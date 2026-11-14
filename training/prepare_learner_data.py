# prepare_learner_data.py – real learner essays as grammar-correction training pairs
#
# Pico's first fine-tuning used Wikipedia-style pairs and synthetic mistakes. On real learner essays (the
# BEA-2019 / CoNLL-2014 / JFLEG benchmarks) it rewrites far too much, because in its training data every
# input had an error. This script converts the human-corrected training sets of the BEA-2019 shared task
# into {"input": ..., "output": ...} lines for training/finetune_grammar.py:
#
#   W&I+LOCNESS train (ABC)   34,308 sentences from learner essays (levels A, B, C)
#   FCE train + dev           30,541 sentences from Cambridge First Certificate exam scripts
#
# About a third of the sentences need no correction; they are kept (input == output), so the model also
# learns to leave correct text alone. Sentences with an error the annotator marked but did not correct
# (UNK edits) are skipped. The test sets used by evaluation/benchmark.py (BEA dev, FCE test, CoNLL-2014,
# JFLEG) are never used here.
#
#   python -m training.prepare_learner_data              # -> data/learner.jsonl
#
# The data is licensed for non-commercial research only: it is downloaded into data/benchmarks/ and
# data/learner.jsonl is git-ignored.

import argparse
import json
import os
import random
import sys
import tarfile
import urllib.request

from evaluation.m2 import corrected_tokens, detok, read_m2

DATA = "data/benchmarks"
SOURCES = [
    ("W&I+LOCNESS train", f"{DATA}/wi+locness/m2/ABC.train.gold.bea19.m2",
     "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz"),
    ("FCE train", f"{DATA}/fce/m2/fce.train.gold.bea19.m2",
     "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz"),
    ("FCE dev", f"{DATA}/fce/m2/fce.dev.gold.bea19.m2",
     "https://www.cl.cam.ac.uk/research/nl/bea2019st/data/fce_v2.1.bea19.tar.gz"),
]

parser = argparse.ArgumentParser(description="Convert BEA-2019 learner corpora into grammar training pairs")
parser.add_argument("--out", default="data/learner.jsonl")
parser.add_argument("--max-words", type=int, default=80, help="skip longer sentences")
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

pairs = []
for name, path, url in SOURCES:
    if not os.path.exists(path):
        archive = os.path.join(DATA, os.path.basename(url))
        os.makedirs(DATA, exist_ok=True)
        if not os.path.exists(archive):
            print(f"downloading {url}")
            urllib.request.urlretrieve(url, archive)
        with tarfile.open(archive) as tar:
            tar.extractall(DATA)
    n = unk = long = same = 0
    for tokens, block in read_m2(path):
        cor = corrected_tokens(tokens, block)
        if cor is None:
            unk += 1
            continue
        src, tgt = detok(tokens)[0], detok(cor)[0]
        if not src or not tgt or len(tokens) > args.max_words:
            long += 1
            continue
        pairs.append({"input": src, "output": tgt})
        n += 1
        same += src == tgt
    print(f"{name:18s} {n:6,} pairs ({same / n:.0%} already correct) | skipped {unk:,} with uncorrected errors, "
          f"{long:,} empty or longer than {args.max_words} words")

random.seed(0)
random.shuffle(pairs)
with open(args.out, "w", encoding="utf-8") as f:
    for p in pairs:
        f.write(json.dumps(p, ensure_ascii=False) + "\n")
same = sum(p["input"] == p["output"] for p in pairs)
print(f"\n{len(pairs):,} pairs ({same:,} already correct, {same / len(pairs):.0%}) -> {args.out}")
for p in pairs[:4]:
    print(f"  {p['input']}\n  -> {p['output']}")
