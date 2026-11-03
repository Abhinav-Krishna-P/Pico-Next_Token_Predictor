import argparse
import hashlib
import math
import os
import random
import time
import urllib.request

import torch
import sentencepiece as spm

from model import Pico


parser = argparse.ArgumentParser(description="Train / run the Pico language model")
parser.add_argument("--retrain", action="store_true", help="train the model from scratch even if pico.pt exists")
parser.add_argument("--resume", action="store_true", help="continue training from pico.pt")
parser.add_argument("--retrain-tokenizer", action="store_true", help="rebuild tokenizer.model from the data files")
parser.add_argument("--steps", type=int, default=None, help="number of training steps (overrides default)")
parser.add_argument("--prompt", type=str, default="Once upon a time", help="text to start generation from")
parser.add_argument("--tokens", type=int, default=120, help="number of tokens to generate")
args = parser.parse_args()

device = "cuda" if torch.cuda.is_available() else "cpu"
print("Torch version:", torch.__version__)
print("Device:", device, "|", torch.cuda.get_device_name(0) if device == "cuda" else f"{os.cpu_count()} CPU threads")

torch.manual_seed(1337)
random.seed(1337)

STORIES_FILE = "data/stories.txt"  # TinyStories: ~27k short stories of simple, grammatical English
STORIES_URL = "https://huggingface.co/datasets/roneneldan/TinyStories/resolve/main/TinyStoriesV2-GPT4-valid.txt"
WORDS_FILE = "words_alpha.txt"      # ~370k real English words, one per line (tokenizer vocabulary only)
CORPUS_FILE = "corpus.txt"          # your own sentences, mixed into the training data
TOKENIZER_PREFIX = "tokenizer"
TOKENS_CACHE = "data/tokens.pt"
CHECKPOINT = "pico.pt"

# ----------------------------
# Hyperparameters
# ----------------------------
# The GPU config is bigger; the CPU config is sized to train in under an hour.
if device == "cuda":
    config = dict(vocab_target=8000, block_size=256, embedding_dim=384, n_heads=6, n_layers=6, dropout=0.1)
    batch_size, max_steps, lr = 64, 5000, 6e-4
else:
    torch.set_num_threads(os.cpu_count())
    config = dict(vocab_target=8000, block_size=64, embedding_dim=192, n_heads=6, n_layers=4, dropout=0.1)
    batch_size, max_steps, lr = 32, 3000, 1e-3
if args.steps is not None:
    max_steps = args.steps
warmup_steps = max(1, max_steps // 20)
min_lr = lr / 10
eval_interval = max(1, max_steps // 10)
eval_iters = 50 if device == "cuda" else 20
log_interval = 10  # print training progress every N steps
weight_decay = 0.1
grad_clip = 1.0


# ----------------------------
# Tokenizer (SentencePiece BPE)
# ----------------------------
def load_lines(path):
    with open(path, "r", encoding="utf-8") as f:
        return [line.strip() for line in f if line.strip()]

def load_stories(path):
    with open(path, "r", encoding="utf-8") as f:
        raw = f.read()
    return [" ".join(s.split()) for s in raw.split("<|endoftext|>") if s.strip()]

if not os.path.exists(STORIES_FILE):
    print(f"Downloading {STORIES_FILE} (~22MB)...")
    os.makedirs(os.path.dirname(STORIES_FILE), exist_ok=True)
    urllib.request.urlretrieve(STORIES_URL, STORIES_FILE)

stories = load_stories(STORIES_FILE)
words = load_lines(WORDS_FILE)
corpus_lines = load_lines(CORPUS_FILE) if os.path.exists(CORPUS_FILE) else []
print(f"Loaded {len(stories):,} stories, {len(words):,} words and {len(corpus_lines)} corpus lines")

if args.retrain_tokenizer or not os.path.exists(TOKENIZER_PREFIX + ".model"):
    print("Training SentencePiece tokenizer...")
    spm.SentencePieceTrainer.train(
        sentence_iterator=iter(stories + words + corpus_lines),
        model_prefix=TOKENIZER_PREFIX,
        vocab_size=config["vocab_target"],
        model_type="bpe",
        character_coverage=1.0,
        input_sentence_size=1_000_000,
        shuffle_input_sentence=True,
        max_sentence_length=20000,  # stories are long "sentences"
        num_threads=os.cpu_count(),
        minloglevel=2,
    )

sp = spm.SentencePieceProcessor()
sp.load(TOKENIZER_PREFIX + ".model")
vocab_size = sp.get_piece_size()
with open(TOKENIZER_PREFIX + ".model", "rb") as f:
    tokenizer_hash = hashlib.md5(f.read()).hexdigest()  # ties checkpoint + token cache to this tokenizer
print("vocab_size :", vocab_size)


# ----------------------------
# Dataset
# ----------------------------
# The model trains on stories (real sentences + grammar). Each story ends with
# an end-of-text token so the model learns where a story stops.
# The word list only shapes the tokenizer's vocabulary: as isolated words in
# random order it carries no grammar.
cache = torch.load(TOKENS_CACHE) if os.path.exists(TOKENS_CACHE) else None
if cache is not None and cache["tokenizer_hash"] == tokenizer_hash and cache["n_docs"] == len(stories) + len(corpus_lines):
    data = cache["data"]
else:
    print("Encoding dataset...")
    docs = stories + corpus_lines
    random.shuffle(docs)
    ids = []
    for piece_ids in sp.encode(docs, out_type=int):
        ids.extend(piece_ids)
        ids.append(sp.eos_id())
    data = torch.tensor(ids, dtype=torch.long)
    torch.save({"tokenizer_hash": tokenizer_hash, "n_docs": len(docs), "data": data}, TOKENS_CACHE)
print(f"Dataset tokens: {len(data):,}")

n_split = int(0.9 * len(data))
train_data, val_data = data[:n_split], data[n_split:]


def get_batch(split, block_size):
    src = train_data if split == "train" else val_data
    ix = torch.randint(len(src) - block_size - 1, (batch_size,))
    x = torch.stack([src[i:i+block_size] for i in ix])
    y = torch.stack([src[i+1:i+block_size+1] for i in ix])
    return x.to(device), y.to(device)


# ----------------------------
# Model
# ----------------------------
@torch.no_grad()
def estimate_loss(model):
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(eval_iters)
        for k in range(eval_iters):
            xb, yb = get_batch(split, model.block_size)
            _, loss = model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


def get_lr(step):
    # linear warmup, then cosine decay down to min_lr
    if step < warmup_steps:
        return lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, max_steps - warmup_steps)
    return min_lr + 0.5 * (lr - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


def make_optimizer(model):
    # no weight decay on biases / LayerNorm weights
    decay = [p for p in model.parameters() if p.dim() >= 2]
    no_decay = [p for p in model.parameters() if p.dim() < 2]
    return torch.optim.AdamW(
        [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
        lr=lr, betas=(0.9, 0.95),
    )


def save_checkpoint(model, optimizer, step, val_loss):
    torch.save({
        "model": model.state_dict(),
        "optimizer": optimizer.state_dict(),
        "config": model_config,
        "step": step,
        "val_loss": val_loss,
    }, CHECKPOINT)


# ----------------------------
# Load or train
# ----------------------------
model_config = {**config, "vocab_size": vocab_size, "tokenizer_hash": tokenizer_hash}
checkpoint = None
if os.path.exists(CHECKPOINT) and not args.retrain:
    ckpt = torch.load(CHECKPOINT, map_location=device)
    if isinstance(ckpt, dict) and ckpt.get("config", {}).get("tokenizer_hash") == tokenizer_hash:
        checkpoint = ckpt
        model_config = ckpt["config"]
    else:
        print(f"{CHECKPOINT} is from an older model / tokenizer, retraining from scratch")

model = Pico(**model_config).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")
optimizer = make_optimizer(model)

start_step = 0
need_training = checkpoint is None or args.resume
if checkpoint is not None:
    model.load_state_dict(checkpoint["model"])
    print(f"Loaded {CHECKPOINT} (step {checkpoint['step']}, val loss {checkpoint['val_loss']:.4f})")
    if args.resume:
        optimizer.load_state_dict(checkpoint["optimizer"])
        start_step = checkpoint["step"]

if need_training:
    best_val = checkpoint["val_loss"] if checkpoint is not None else float("inf")
    end_step = start_step + max_steps
    t0 = time.time()
    model.train()
    for step in range(start_step, end_step):
        for group in optimizer.param_groups:
            group["lr"] = get_lr(step - start_step)  # schedule restarts on each (resumed) run

        if (step - start_step) % eval_interval == 0 or step == end_step - 1:
            losses = estimate_loss(model)
            elapsed = time.time() - t0
            print(f"Step {step:5d} | train {losses['train']:.4f} | val {losses['val']:.4f} "
                  f"| lr {optimizer.param_groups[0]['lr']:.2e} | {elapsed:.0f}s", flush=True)
            if losses["val"] < best_val:
                best_val = losses["val"]
                save_checkpoint(model, optimizer, step, best_val)

        xb, yb = get_batch("train", model.block_size)
        _, loss = model(xb, yb)
        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
        optimizer.step()

        done = step - start_step + 1
        if done % log_interval == 0:
            elapsed = time.time() - t0
            eta = elapsed / done * (end_step - step - 1)
            print(f"  step {step + 1:5d}/{end_step} | loss {loss.item():.4f} | "
                  f"{elapsed / done:.2f}s/step | elapsed {elapsed / 60:.1f}m | ETA {eta / 60:.1f}m", flush=True)

    print(f"Training done in {time.time() - t0:.0f}s, best val loss {best_val:.4f} saved to {CHECKPOINT}")
    model.load_state_dict(torch.load(CHECKPOINT, map_location=device)["model"])


# ----------------------------
# Generate
# ----------------------------
model.eval()
context = torch.tensor([sp.encode(args.prompt, out_type=int)], dtype=torch.long, device=device)
out = model.generate(context, max_new_tokens=args.tokens)[0].tolist()
print("Generated text:", sp.decode(out))
