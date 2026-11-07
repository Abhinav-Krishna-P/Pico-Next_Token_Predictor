# correct.py – fix grammar with a fine-tuned model
#
#   python correct.py "they was playing in park"                          # correct one sentence
#   python correct.py                                                     # interactive: empty line to quit
#   python correct.py --model my_model.pt "she go to school"             # use another model (looks in models/ too)

import argparse
import os
import sys

import torch

from pico.grammar import correct, load_grammar_model, resolve_model_path

parser = argparse.ArgumentParser(description="Correct grammar with Pico")
parser.add_argument("--model", default="models/pico_big_grammar_small.pt", help="fine-tuned checkpoint")
parser.add_argument("sentence", nargs="*", help="sentence to correct (omit for interactive mode)")
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")  # Windows consoles default to cp1252

args.model = resolve_model_path(args.model)
if not os.path.exists(args.model):
    sys.exit(f"{args.model} not found (also looked in models/): train the grammar model first with  "
             f"python -m training.finetune_grammar")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, sp, _ = load_grammar_model(args.model, device)

if args.sentence:
    print(correct(model, sp, " ".join(args.sentence))[0])
else:
    while True:
        sentence = input("Sentence: ")
        if not sentence.strip():
            break
        print("Corrected:", correct(model, sp, sentence)[0])
