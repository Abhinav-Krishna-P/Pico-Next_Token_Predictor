# grammar.py – load a fine-tuned grammar model and correct sentences (used by correct.py and the evaluation)

import os

import torch
import sentencepiece as spm

from .model import Pico, find_tokenizer

PROMPT = "Incorrect: {} Correct:"   # the format finetune_grammar.py trains on


def resolve_model_path(path):
    """Accept a bare file name for models in models/ (e.g. "pico_grammar.pt")."""
    if not os.path.exists(path) and os.path.exists(os.path.join("models", path)):
        return os.path.join("models", path)
    return path


def load_grammar_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    sp = spm.SentencePieceProcessor(model_file=find_tokenizer(ckpt, path, "models/tokenizer_grammar.model"))
    model = Pico(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, sp, ckpt


@torch.no_grad()
def correct(model, sp, sentence):
    """Greedy-decode the corrected sentence. Returns (text, prompt_tokens, generated_tokens)."""
    prompt = sp.encode(PROMPT.format(sentence.strip()), out_type=int)
    max_new = model.block_size - len(prompt)
    if max_new <= 0:
        return "(sentence too long for the model's context)", len(prompt), 0
    device = next(model.parameters()).device
    idx = torch.tensor([prompt], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new, temperature=0, stop_id=sp.eos_id())
    new = out[0, len(prompt):].tolist()
    return sp.decode(new), len(prompt), len(new)
