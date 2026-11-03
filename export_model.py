# export_model.py – make a smaller checkpoint for using the model (not training it)
#
# Training checkpoints also store the optimizer state, which roughly triples their size
# and is only needed to continue training. This keeps just the weights + config.
#
#   python export_model.py pico_big_grammar.pt pico_big_grammar_small.pt

import os
import sys

import torch

if len(sys.argv) != 3:
    sys.exit("usage: python export_model.py <checkpoint.pt> <output.pt>")
src, dst = sys.argv[1], sys.argv[2]

ckpt = torch.load(src, map_location="cpu")
small = {k: ckpt[k] for k in ("model", "config", "step", "val_loss", "tokenizer") if k in ckpt}
torch.save(small, dst)
print(f"{src} ({os.path.getsize(src) / 1e6:.0f}MB) -> {dst} ({os.path.getsize(dst) / 1e6:.0f}MB)")
