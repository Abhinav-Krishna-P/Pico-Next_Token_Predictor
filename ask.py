# ask.py – ask Pico a grammar question (models fine-tuned with the English Practical Grammar dataset)
#
#   python ask.py "What is a noun?"
#   python ask.py                         # interactive: empty line to quit

import argparse
import os
import sys

import torch

from pico.grammar import answer, load_grammar_model, resolve_model_path

parser = argparse.ArgumentParser(description="Ask Pico a grammar question")
parser.add_argument("--model", default="models/pico_big_grammar_small.pt", help="fine-tuned checkpoint")
parser.add_argument("question", nargs="*", help="question (omit for interactive mode)")
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")

args.model = resolve_model_path(args.model)
if not os.path.exists(args.model):
    sys.exit(f"{args.model} not found (also looked in models/)")

device = "cuda" if torch.cuda.is_available() else "cpu"
model, sp, _ = load_grammar_model(args.model, device)

if args.question:
    print(answer(model, sp, " ".join(args.question))[0])
else:
    while True:
        question = input("Question: ")
        if not question.strip():
            break
        print(answer(model, sp, question)[0], "\n")
