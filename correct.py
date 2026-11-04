# correct.py – fix grammar with a fine-tuned model
#
#   python correct.py "they was playing in park"                          # correct one sentence
#   python correct.py                                                     # interactive: empty line to quit
#   python correct.py --model pico_big_grammar.pt "she go to school"      # use another model

import argparse
import os
import sys

import torch
import sentencepiece as spm

from pico import Pico, find_tokenizer

parser = argparse.ArgumentParser(description="Correct grammar with Pico")
parser.add_argument("--model", default="models/pico_big_grammar_small.pt", help="fine-tuned checkpoint")
parser.add_argument("sentence", nargs="*", help="sentence to correct (omit for interactive mode)")
args = parser.parse_args()

PROMPT = "Incorrect: {} Correct:"
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

if not os.path.exists(args.model):
    sys.exit(f"{args.model} not found: train the grammar model first with  python -m training.finetune_grammar")

device = "cuda" if torch.cuda.is_available() else "cpu"
ckpt = torch.load(args.model, map_location=device)
sp = spm.SentencePieceProcessor(model_file=find_tokenizer(ckpt, args.model, "models/tokenizer_grammar.model"))
model = Pico(**ckpt["config"]).to(device)
model.load_state_dict(ckpt["model"])
model.eval()


def correct(sentence):
    prompt = sp.encode(PROMPT.format(sentence.strip()), out_type=int)
    max_new = model.block_size - len(prompt)
    if max_new <= 0:
        return "(sentence too long for the model's context)"
    idx = torch.tensor([prompt], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new, temperature=0, stop_id=sp.eos_id())
    return sp.decode(out[0, len(prompt):].tolist())


if args.sentence:
    print(correct(" ".join(args.sentence)))
else:
    while True:
        sentence = input("Sentence: ")
        if not sentence.strip():
            break
        print("Corrected:", correct(sentence))
