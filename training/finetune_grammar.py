# finetune_grammar.py – fine-tune the pre-trained Pico (pico.pt) to correct grammar
#
# Data: .jsonl files with one example per line, either
#   {"input": "<wrong sentence>", "output": "<corrected sentence>"}  ->  Incorrect: <input> Correct: <output></s>
#   {"messages": [{"role": "user", ...}, {"role": "assistant", ...}]} ->  Question: <user> Answer: <assistant></s>
# The loss is computed only on the part after "Correct:" / "Answer:", so the model learns to write the
# correction or answer (not to predict the mistakes). Add "*N" to a file to repeat it N times in training,
# e.g. --data data/train.jsonl "data/grammar_book.jsonl*10".

import argparse
import json
import math
import os
import random
import sys
import time
from contextlib import nullcontext

import torch
import sentencepiece as spm
from sentencepiece import sentencepiece_model_pb2 as sp_pb2

from pico import Pico, find_tokenizer
from pico.grammar import PROMPT, QA_PROMPT


parser = argparse.ArgumentParser(description="Fine-tune Pico for grammar correction")
parser.add_argument("--data", nargs="+", default=["data/train.jsonl"],
                    help="jsonl files with input/output pairs or chat messages; 'file*N' repeats a file N times")
parser.add_argument("--base", default="models/pico.pt", help="pre-trained checkpoint to start from")
parser.add_argument("--out", default="models/pico_grammar.pt", help="where to save the fine-tuned model")
parser.add_argument("--resume", action="store_true", help="continue fine-tuning from --out")
parser.add_argument("--steps", type=int, default=3000, help="number of training steps")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--tokenizer-out", default="models/tokenizer_grammar.model", help="where to save the fine-tuning tokenizer")
parser.add_argument("--block-size", type=int, default=None,
                    help="max tokens per example (default: the base model's context, at least 128)")
parser.add_argument("--lr", type=float, default=3e-4)
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cpu":
    torch.set_num_threads(os.cpu_count())
    amp_dtype = None
else:
    # bfloat16 on Ampere+, float16 + loss scaling on older GPUs (T4)
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
autocast = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
print("Device:", device, "| precision:", amp_dtype or "float32")
torch.manual_seed(1337)
random.seed(1337)

if args.resume:
    ckpt = torch.load(args.out, map_location=device)
    print(f"Resuming from {args.out} (step {ckpt['step']}, val loss {ckpt['val_loss']:.4f})")
else:
    ckpt = torch.load(args.base, map_location=device)
    print(f"Starting from pre-trained {args.base} (val loss {ckpt['val_loss']:.4f})")
# pretrain.py stores its tokenizer path in the checkpoint; the small train_small.py model uses models/tokenizer.model
BASE_TOKENIZER = find_tokenizer(ckpt, args.base, "models/tokenizer.model")
# when resuming, keep using the tokenizer the run was started with
TOKENIZER = find_tokenizer(ckpt, args.out, args.tokenizer_out) if args.resume else args.tokenizer_out
if args.block_size is None:
    args.block_size = max(128, ckpt["config"]["block_size"])
warmup_steps = max(1, args.steps // 20)
min_lr = args.lr / 10
eval_interval = max(1, args.steps // 10)
log_interval = 10
weight_decay = 0.1
grad_clip = 1.0

rows = []   # (prompt text, answer text, times repeated in training)
for spec in args.data:
    path, _, repeat = spec.rpartition("*") if spec.rpartition("*")[2].isdigit() else (spec, "", "1")
    n = len(rows)
    for line in open(path, encoding="utf-8"):
        if not line.strip():
            continue
        r = json.loads(line)
        if "messages" in r:        # chat example: every user turn with the assistant reply after it
            msgs = r["messages"]
            rows += [(QA_PROMPT.format(u["content"]), a["content"], int(repeat))
                     for u, a in zip(msgs, msgs[1:]) if u["role"] == "user" and a["role"] == "assistant"]
        else:
            rows.append((PROMPT.format(r["input"]), r["output"], int(repeat)))
    print(f"Loaded {len(rows) - n:,} examples from {path}" + (f" (x{repeat} in training)" if repeat != "1" else ""))


# ----------------------------
# Tokenizer: base tokenizer + missing symbols
# ----------------------------
# The base tokenizer only saw children's stories, so symbols like ( ) & $ % are
# unknown to it. They are appended as new tokens at the END of the vocabulary,
# so every existing token keeps its id and the pre-trained weights still match.
if not args.resume:
    base_sp = spm.SentencePieceProcessor(model_file=BASE_TOKENIZER)
    counts = {}
    for prompt_text, answer_text, _ in rows:
        for ch in set(prompt_text + answer_text):
            if not ch.isspace() and ch != "�" and base_sp.unk_id() in base_sp.encode(ch):
                counts[ch] = counts.get(ch, 0) + 1
    new_symbols = sorted(ch for ch, n in counts.items() if n >= 20)  # ignore very rare characters

    proto = sp_pb2.ModelProto()
    with open(BASE_TOKENIZER, "rb") as f:
        proto.ParseFromString(f.read())
    for ch in new_symbols:
        piece = proto.pieces.add()
        piece.piece = ch
        piece.score = 0.0
        piece.type = sp_pb2.ModelProto.SentencePiece.USER_DEFINED
    with open(TOKENIZER, "wb") as f:
        f.write(proto.SerializeToString())
    print(f"Added {len(new_symbols)} symbols to the tokenizer: {''.join(new_symbols)}")

sp = spm.SentencePieceProcessor(model_file=TOKENIZER)
vocab_size = sp.get_piece_size()
eos_id = sp.eos_id()
print("vocab_size :", vocab_size)


# ----------------------------
# Dataset
# ----------------------------
examples = []
skipped_long = skipped_unk = 0
prompts = sp.encode([r[0] for r in rows], out_type=int)
answers = sp.encode([r[1] for r in rows], out_type=int)
for p, a, (_, _, repeat) in zip(prompts, answers, rows):
    if sp.unk_id() in a:
        skipped_unk += 1       # never teach the model to output <unk>
        continue
    a = a + [eos_id]           # end-of-text marks where the correction stops
    if len(p) + len(a) > args.block_size + 1:
        skipped_long += 1
        continue
    examples.append((p, a, repeat))
print(f"Using {len(examples):,} examples (skipped {skipped_long:,} too long for {args.block_size} tokens, "
      f"{skipped_unk:,} with unknown characters)")

random.shuffle(examples)
n_val = min(1000, len(examples) // 20)
val_examples = [(p, a) for p, a, _ in examples[:n_val]]
train_examples = [(p, a) for p, a, repeat in examples[n_val:] for _ in range(repeat)]   # repeats after the split
print(f"{len(train_examples):,} training / {len(val_examples):,} validation examples")


def make_batch(batch):
    # Pad every example to the longest in the batch. Prompt and padding targets are
    # -100 so they don't count towards the loss.
    T = max(len(p) + len(a) for p, a in batch) - 1
    x = torch.full((len(batch), T), eos_id, dtype=torch.long)
    y = torch.full((len(batch), T), -100, dtype=torch.long)
    for i, (p, a) in enumerate(batch):
        seq = torch.tensor(p + a, dtype=torch.long)
        n = len(seq) - 1
        x[i, :n] = seq[:-1]
        y[i, :n] = seq[1:]
        y[i, :len(p) - 1] = -100   # don't train on predicting the (incorrect) prompt
    return x.to(device), y.to(device)


def get_batch(split):
    src = train_examples if split == "train" else val_examples
    return make_batch(random.sample(src, args.batch_size))


# ----------------------------
# Model
# ----------------------------
model_config = ckpt["config"]
model = Pico(**model_config).to(device)
model.load_state_dict(ckpt["model"])
model.resize(vocab_size=vocab_size, block_size=args.block_size)   # new symbols + longer context
model_config = {**model_config, "vocab_size": vocab_size, "block_size": model.block_size}
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,} | context {model.block_size} tokens")

decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": weight_decay}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95),
)
scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))
start_step = 0
best_val = float("inf")
if args.resume:
    optimizer.load_state_dict(ckpt["optimizer"])
    start_step = ckpt["step"]
    best_val = ckpt["val_loss"]


def get_lr(step):
    # linear warmup, then cosine decay down to min_lr
    if step < warmup_steps:
        return args.lr * (step + 1) / warmup_steps
    progress = (step - warmup_steps) / max(1, args.steps - warmup_steps)
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def estimate_val_loss():
    model.eval()
    losses = []
    for i in range(0, len(val_examples), args.batch_size):
        with autocast:
            _, loss = model(*make_batch(val_examples[i:i + args.batch_size]))
        losses.append(loss.item())
    model.train()
    return sum(losses) / len(losses)


@torch.no_grad()
def correct(prompt_ids):
    model.eval()
    idx = torch.tensor([prompt_ids], dtype=torch.long, device=device)
    with autocast:
        out = model.generate(idx, max_new_tokens=model.block_size - len(prompt_ids), temperature=0, stop_id=eos_id)
    model.train()
    return sp.decode(out[0, len(prompt_ids):].tolist())


# ----------------------------
# Fine-tune
# ----------------------------
end_step = start_step + args.steps
t0 = time.time()
model.train()
for step in range(start_step, end_step):
    for group in optimizer.param_groups:
        group["lr"] = get_lr(step - start_step)

    if (step - start_step) % eval_interval == 0 or step == end_step - 1:
        val_loss = estimate_val_loss()
        print(f"Step {step:5d} | val {val_loss:.4f} | lr {optimizer.param_groups[0]['lr']:.2e} "
              f"| {time.time() - t0:.0f}s", flush=True)
        p, a = val_examples[0]
        print(f"    {sp.decode(p)}\n    -> {correct(p)}", flush=True)
        if val_loss < best_val:
            best_val = val_loss
            torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(),
                        "config": model_config, "step": step, "val_loss": val_loss, "tokenizer": TOKENIZER}, args.out)

    xb, yb = get_batch("train")
    with autocast:
        _, loss = model(xb, yb)
    optimizer.zero_grad(set_to_none=True)
    scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), grad_clip)
    scaler.step(optimizer)
    scaler.update()

    done = step - start_step + 1
    if done % log_interval == 0:
        elapsed = time.time() - t0
        eta = elapsed / done * (end_step - step - 1)
        print(f"  step {step + 1:5d}/{end_step} | loss {loss.item():.4f} | "
              f"{elapsed / done:.2f}s/step | elapsed {elapsed / 60:.1f}m | ETA {eta / 60:.1f}m", flush=True)

print(f"Fine-tuning done in {time.time() - t0:.0f}s, best val loss {best_val:.4f} saved to {args.out}")
print(f"Try it: python correct.py --model {args.out} \"they was playing in park\"")
