# pretrain.py – large-scale pre-training of Pico on the data from prepare_data.py (made for a GPU / Colab)
#
#   python -m training.pretrain --data-dir data/big --out models/pico_big.pt
#
# Automatically resumes from --out if it exists (so a Colab disconnect only loses a few
# minutes): just run the same command again. Use --fresh to start over.

import argparse
import math
import os
import shutil
import sys
import time
from contextlib import nullcontext

import numpy as np
import torch
import sentencepiece as spm

from pico import Pico

parser = argparse.ArgumentParser(description="Pre-train Pico on a large dataset")
parser.add_argument("--data-dir", default="data/big")
parser.add_argument("--out", default="models/pico_big.pt")
parser.add_argument("--fresh", action="store_true", help="ignore an existing checkpoint and start from scratch")
# model (~34M parameters with the defaults)
parser.add_argument("--block-size", type=int, default=512)
parser.add_argument("--embedding-dim", type=int, default=512)
parser.add_argument("--n-heads", type=int, default=8)
parser.add_argument("--n-layers", type=int, default=8)
parser.add_argument("--dropout", type=float, default=0.0, help="not needed when every token is seen ~once")
# training
parser.add_argument("--batch-size", type=int, default=32, help="sequences per micro-batch")
parser.add_argument("--grad-accum", type=int, default=8, help="micro-batches per optimizer step")
parser.add_argument("--max-tokens", type=float, default=600e6, help="total training tokens")
parser.add_argument("--lr", type=float, default=1e-3)
parser.add_argument("--warmup-steps", type=int, default=200)
parser.add_argument("--eval-interval", type=int, default=250)
parser.add_argument("--eval-iters", type=int, default=40)
parser.add_argument("--save-minutes", type=float, default=10, help="also save a checkpoint every N minutes")
parser.add_argument("--compile", action="store_true", help="torch.compile the model (faster on modern GPUs)")
parser.add_argument("--time-limit-hours", type=float, default=None,
                    help="save and stop after this long (e.g. before a Kaggle/Colab session ends); re-run to resume")
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

device = "cuda" if torch.cuda.is_available() else "cpu"
if device == "cuda":
    # bfloat16 on Ampere+ (A100, L4), float16 + loss scaling on older GPUs (T4)
    amp_dtype = torch.bfloat16 if torch.cuda.get_device_capability()[0] >= 8 else torch.float16
    torch.backends.cuda.matmul.allow_tf32 = True
    torch.backends.cudnn.allow_tf32 = True
    print("GPU:", torch.cuda.get_device_name(0), "| precision:", amp_dtype)
else:
    amp_dtype = None
    torch.set_num_threads(os.cpu_count())
    print("WARNING: no GPU found, this will be very slow (use it only for testing)")
autocast = torch.autocast(device_type="cuda", dtype=amp_dtype) if amp_dtype else nullcontext()
torch.manual_seed(1337)

# keep a copy of the tokenizer next to the checkpoint, so the two always travel together
tokenizer_path = os.path.splitext(args.out)[0] + "_tokenizer.model"
shutil.copyfile(os.path.join(args.data_dir, "tokenizer.model"), tokenizer_path)
sp = spm.SentencePieceProcessor(model_file=tokenizer_path)
vocab_size = sp.get_piece_size()

tokens_per_step = args.batch_size * args.grad_accum * args.block_size
max_steps = int(args.max_tokens // tokens_per_step)
min_lr = args.lr / 10


# ----------------------------
# Data (memory-mapped, never fully loaded into RAM)
# ----------------------------
def get_batch(split):
    # re-open the memmap every call: avoids a known memory leak with long-lived memmaps
    data = np.memmap(os.path.join(args.data_dir, f"{split}.bin"), dtype=np.uint16, mode="r")
    ix = torch.randint(len(data) - args.block_size - 1, (args.batch_size,))
    x = torch.from_numpy(np.stack([data[i:i + args.block_size].astype(np.int64) for i in ix]))
    y = torch.from_numpy(np.stack([data[i + 1:i + 1 + args.block_size].astype(np.int64) for i in ix]))
    if device == "cuda":
        return x.pin_memory().to(device, non_blocking=True), y.pin_memory().to(device, non_blocking=True)
    return x, y

n_train = os.path.getsize(os.path.join(args.data_dir, "train.bin")) // 2
print(f"Train tokens: {n_train:,} | {tokens_per_step:,} tokens/step | {max_steps:,} steps "
      f"({args.max_tokens / n_train:.2f} epochs)")


# ----------------------------
# Model / optimizer / resume
# ----------------------------
ckpt = None
if os.path.exists(args.out) and not args.fresh:
    ckpt = torch.load(args.out, map_location=device)
    model_config = ckpt["config"]
    print(f"Resuming from {args.out} (step {ckpt['step']}, val loss {ckpt['val_loss']:.4f})")
else:
    model_config = dict(vocab_size=vocab_size, block_size=args.block_size, embedding_dim=args.embedding_dim,
                        n_heads=args.n_heads, n_layers=args.n_layers, dropout=args.dropout)
model = Pico(**model_config).to(device)
print(f"Parameters: {sum(p.numel() for p in model.parameters()):,}")

decay = [p for p in model.parameters() if p.dim() >= 2]
no_decay = [p for p in model.parameters() if p.dim() < 2]
optimizer = torch.optim.AdamW(
    [{"params": decay, "weight_decay": 0.1}, {"params": no_decay, "weight_decay": 0.0}],
    lr=args.lr, betas=(0.9, 0.95), fused=(device == "cuda"),
)
scaler = torch.amp.GradScaler("cuda", enabled=(amp_dtype == torch.float16))

start_step, best_val = 0, float("inf")
if ckpt is not None:
    model.load_state_dict(ckpt["model"])
    optimizer.load_state_dict(ckpt["optimizer"])
    if "scaler" in ckpt:
        scaler.load_state_dict(ckpt["scaler"])
    start_step, best_val = ckpt["step"] + 1, ckpt["val_loss"]
    del ckpt

train_model = torch.compile(model) if args.compile else model


def get_lr(step):
    # linear warmup, then cosine decay down to min_lr
    if step < args.warmup_steps:
        return args.lr * (step + 1) / args.warmup_steps
    progress = (step - args.warmup_steps) / max(1, max_steps - args.warmup_steps)
    return min_lr + 0.5 * (args.lr - min_lr) * (1 + math.cos(math.pi * min(1.0, progress)))


@torch.no_grad()
def estimate_loss():
    model.eval()
    out = {}
    for split in ("train", "val"):
        losses = torch.zeros(args.eval_iters)
        for k in range(args.eval_iters):
            xb, yb = get_batch(split)
            with autocast:
                _, loss = train_model(xb, yb)
            losses[k] = loss.item()
        out[split] = losses.mean().item()
    model.train()
    return out


@torch.no_grad()
def sample(prompt, tokens=60):
    model.eval()
    idx = torch.tensor([sp.encode(prompt)], dtype=torch.long, device=device)
    with autocast:
        out = model.generate(idx, max_new_tokens=tokens, stop_id=sp.eos_id())
    model.train()
    return sp.decode(out[0].tolist())


def save_checkpoint(step, val_loss, finished=False):
    # write to a temp file first so a disconnect mid-save can't corrupt the checkpoint
    tmp = args.out + ".tmp"
    torch.save({"model": model.state_dict(), "optimizer": optimizer.state_dict(), "scaler": scaler.state_dict(),
                "config": model_config, "step": step, "val_loss": val_loss, "tokenizer": tokenizer_path,
                "finished": finished}, tmp)
    shutil.move(tmp, args.out)


# ----------------------------
# Train
# ----------------------------
if start_step >= max_steps:
    print(f"Already trained for {max_steps} steps. Raise --max-tokens to train longer.")
t0 = last_save = time.time()
last_val = best_val
model.train()
for step in range(start_step, max_steps):
    lr = get_lr(step)
    for group in optimizer.param_groups:
        group["lr"] = lr

    if step % args.eval_interval == 0 and step > start_step or step == max_steps - 1:
        losses = estimate_loss()
        last_val = losses["val"]
        best_val = min(best_val, last_val)
        print(f"Step {step:6d} | train {losses['train']:.4f} | val {losses['val']:.4f} | lr {lr:.2e}", flush=True)
        for prompt in ("Once upon a time", "The scientist explained that"):
            print(f"    {sample(prompt)}", flush=True)
        save_checkpoint(step, last_val)
        last_save = time.time()

    for micro in range(args.grad_accum):
        xb, yb = get_batch("train")
        with autocast:
            _, loss = train_model(xb, yb)
            loss = loss / args.grad_accum
        scaler.scale(loss).backward()
    scaler.unscale_(optimizer)
    torch.nn.utils.clip_grad_norm_(model.parameters(), 1.0)
    scaler.step(optimizer)
    scaler.update()
    optimizer.zero_grad(set_to_none=True)

    done = step - start_step + 1
    if done % 10 == 0:
        elapsed = time.time() - t0
        tok_s = done * tokens_per_step / elapsed
        eta = (max_steps - step - 1) * elapsed / done
        print(f"  step {step + 1:6d}/{max_steps} | loss {loss.item() * args.grad_accum:.4f} | "
              f"{tok_s:,.0f} tok/s | elapsed {elapsed / 60:.1f}m | ETA {eta / 60:.1f}m", flush=True)

    if time.time() - last_save > args.save_minutes * 60:
        save_checkpoint(step, last_val)
        last_save = time.time()

    if args.time_limit_hours is not None and time.time() - t0 > args.time_limit_hours * 3600:
        save_checkpoint(step, last_val)
        print(f"Time limit reached at step {step + 1}/{max_steps}, saved to {args.out}. "
              f"Run again to continue.", flush=True)
        break
else:
    save_checkpoint(max_steps - 1, last_val, finished=True)
    print(f"Pre-training done, val loss {last_val:.4f}, saved to {args.out}")
