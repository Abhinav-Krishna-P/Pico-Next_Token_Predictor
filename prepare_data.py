# prepare_data.py – build the large pre-training dataset (run once, e.g. on Google Colab)
#
# 1. Downloads full TinyStories (fluent, simple grammar) and FineWeb-Edu (educational
#    web text for adult vocabulary) from Hugging Face.
# 2. Trains a 16k SentencePiece tokenizer with byte fallback (no character is ever <unk>).
# 3. Tokenizes everything into train.bin / val.bin (uint16 token ids, read with np.memmap
#    so gigabytes of data never have to fit in RAM).
#
#   python prepare_data.py --out-dir data_big                 # full size (~800M tokens)
#   python prepare_data.py --out-dir data_test --stories-docs 2000 --web-docs 500 --vocab-size 2000   # quick test

import argparse
import json
import os
import random

import numpy as np
import sentencepiece as spm
from datasets import load_dataset

parser = argparse.ArgumentParser(description="Download and tokenize Pico's pre-training data")
parser.add_argument("--out-dir", default="data_big")
parser.add_argument("--stories-docs", type=int, default=None, help="max TinyStories documents (default: all ~2.1M)")
parser.add_argument("--web-docs", type=int, default=300_000, help="FineWeb-Edu documents (~1k tokens each)")
parser.add_argument("--vocab-size", type=int, default=16384)
parser.add_argument("--tokenizer-docs", type=int, default=300_000, help="documents sampled to train the tokenizer")
parser.add_argument("--val-fraction", type=float, default=0.005)
args = parser.parse_args()

os.makedirs(args.out_dir, exist_ok=True)
random.seed(1337)
raw_files = {
    "stories": os.path.join(args.out_dir, "raw_stories.txt"),
    "web": os.path.join(args.out_dir, "raw_web.txt"),
}
tokenizer_prefix = os.path.join(args.out_dir, "tokenizer")


# ----------------------------
# 1. Download (one document per line)
# ----------------------------
def download(name, dataset_args, max_docs, path):
    if os.path.exists(path):
        print(f"{path} already exists, skipping download")
        return
    print(f"Downloading {name}...")
    ds = load_dataset(*dataset_args, split="train", streaming=True)
    n = 0
    with open(path + ".tmp", "w", encoding="utf-8") as f:
        for row in ds:
            text = " ".join(row["text"].split())   # one line per document
            if not text:
                continue
            f.write(text + "\n")
            n += 1
            if n % 100_000 == 0:
                print(f"  {n:,} documents", flush=True)
            if max_docs is not None and n >= max_docs:
                break
    os.replace(path + ".tmp", path)
    print(f"  {n:,} documents saved to {path}")

download("TinyStories", ("roneneldan/TinyStories",), args.stories_docs, raw_files["stories"])
download("FineWeb-Edu", ("HuggingFaceFW/fineweb-edu", "sample-10BT"), args.web_docs, raw_files["web"])


# ----------------------------
# 2. Tokenizer
# ----------------------------
if not os.path.exists(tokenizer_prefix + ".model"):
    print("Training tokenizer...")
    inputs = list(raw_files.values())
    if os.path.exists("words_alpha.txt"):
        inputs.append("words_alpha.txt")   # extra coverage for rare English words
    spm.SentencePieceTrainer.train(
        input=inputs,
        model_prefix=tokenizer_prefix,
        vocab_size=args.vocab_size,
        model_type="bpe",
        byte_fallback=True,            # unknown characters become UTF-8 bytes instead of <unk>
        character_coverage=0.9995,
        input_sentence_size=args.tokenizer_docs,
        shuffle_input_sentence=True,
        max_sentence_length=32768,
        num_threads=os.cpu_count(),
        minloglevel=2,
    )
sp = spm.SentencePieceProcessor(model_file=tokenizer_prefix + ".model")
assert sp.get_piece_size() < 2**16, "token ids must fit in uint16"
print("vocab_size :", sp.get_piece_size())


# ----------------------------
# 3. Tokenize into train.bin / val.bin
# ----------------------------
def encode_file(path, train_f, val_f, chunk_docs=20_000):
    counts = {"train": 0, "val": 0}
    with open(path, encoding="utf-8") as f:
        while True:
            docs = [line.rstrip("\n") for _, line in zip(range(chunk_docs), f)]
            if not docs:
                break
            for ids in sp.encode(docs, out_type=int, num_threads=os.cpu_count()):
                ids.append(sp.eos_id())   # end-of-text between documents
                split = "val" if random.random() < args.val_fraction else "train"
                np.array(ids, dtype=np.uint16).tofile(val_f if split == "val" else train_f)
                counts[split] += len(ids)
            print(f"  {path}: {counts['train'] + counts['val']:,} tokens", flush=True)
    return counts

train_path = os.path.join(args.out_dir, "train.bin")
val_path = os.path.join(args.out_dir, "val.bin")
if not os.path.exists(train_path):
    print("Tokenizing...")
    meta = {"vocab_size": sp.get_piece_size(), "sources": {}}
    with open(train_path + ".tmp", "wb") as train_f, open(val_path, "wb") as val_f:
        for name, path in raw_files.items():
            meta["sources"][name] = encode_file(path, train_f, val_f)
    os.replace(train_path + ".tmp", train_path)
    with open(os.path.join(args.out_dir, "meta.json"), "w") as f:
        json.dump(meta, f, indent=2)

n_train = os.path.getsize(train_path) // 2
n_val = os.path.getsize(val_path) // 2
print(f"Done: {n_train:,} train tokens, {n_val:,} val tokens in {args.out_dir}/")
