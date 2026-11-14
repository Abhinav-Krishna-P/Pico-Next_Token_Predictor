# grammar.py – load a fine-tuned grammar model and correct sentences (used by correct.py and the evaluation)

import difflib
import os
import re

import torch
import torch.nn.functional as F
import sentencepiece as spm

from .model import Pico, find_tokenizer

PROMPT = "Incorrect: {} Correct:"   # the formats finetune_grammar.py trains on: sentence correction
QA_PROMPT = "Question: {} Answer:"  # and grammar questions (English Practical Grammar dataset)


def resolve_model_path(path):
    """Accept a bare file name for models in models/ (e.g. "pico_big_grammar_small.pt")."""
    if not os.path.exists(path) and os.path.exists(os.path.join("models", path)):
        return os.path.join("models", path)
    return path


def load_grammar_model(path, device="cpu"):
    ckpt = torch.load(path, map_location=device)
    sp = spm.SentencePieceProcessor(model_file=find_tokenizer(ckpt, path, "models/pico_big_grammar_tokenizer.model"))
    model = Pico(**ckpt["config"]).to(device)
    model.load_state_dict(ckpt["model"])
    model.eval()
    return model, sp, ckpt


def correct(model, sp, sentence, filter=True):
    """Greedy-decode the corrected sentence, then (filter=True) keep only the edits the filter trusts
    (see filter_correction below). Returns (text, prompt_tokens, generated_tokens)."""
    sentence = " ".join(sentence.split())
    text, n_prompt, n_new = complete(model, sp, PROMPT.format(sentence))
    if filter:
        text = filter_correction(model, sp, sentence, text)
    return text, n_prompt, n_new


def answer(model, sp, question):
    """Answer a grammar question (models fine-tuned with the chat dataset)."""
    return complete(model, sp, QA_PROMPT.format(question.strip()))


@torch.no_grad()
def complete(model, sp, prompt_text):
    prompt = sp.encode(prompt_text, out_type=int)
    max_new = model.block_size - len(prompt)
    if max_new <= 0:
        return "(sentence too long for the model's context)", len(prompt), 0
    device = next(model.parameters()).device
    idx = torch.tensor([prompt], dtype=torch.long, device=device)
    out = model.generate(idx, max_new_tokens=max_new, temperature=0, stop_id=sp.eos_id())
    new = out[0, len(prompt):].tolist()
    return sp.decode(new), len(prompt), len(new)


# ----------------------------
# Edit filter
# ----------------------------
# On real learner essays (BEA-2019, CoNLL-2014, JFLEG – see evaluation/benchmark.py) most of Pico's edits are
# wrong. The correction is therefore split into word-level edits, and an edit is kept only if
#   1. it is not a "rewrite": it changes only punctuation / capitalization, only small function words
#      (articles, prepositions, auxiliaries, pronouns, ...), or one word into a similar one (go -> goes,
#      recieve -> receive). Rewrites of a word into an unrelated one ("Psychology" -> "the parts") were 50%
#      of Pico's edits on BEA dev, and only 2% of them were right.
#   2. the model is not clearly against it: log P(input with only this edit) - log P(input unchanged)
#      must be above MIN_GAIN.
# Tuned on half of BEA-2019 dev; on the other half F0.5 went from 0.10 to 0.17, on CoNLL-2014 from 0.10 to
# 0.18 and on JFLEG from 0.26 to 0.36.
MIN_GAIN = -2.0
FUNCTION_WORDS = set("""a an the this that these those some any no every each another my your his her its our their
in on at to for of with from by about into onto over under between among through during before after since until
without within along across behind beyond near towards toward upon off out up down around than as like
is am are was were be been being do does did have has had will would shall should can could may might must
i me you he him she it we us they them myself yourself himself herself itself ourselves themselves
and or but so because if when while although though whether which who whom whose what where how why
not there here very too also much many more most less few little all both either neither such""".split())


IRREGULAR = {form: base for line in """
be was were been being am is are | go went gone goes going | do did done does doing | have had has having
buy bought | bring brought | build built | catch caught | choose chose chosen | come came | cut | dig dug
draw drew drawn | drink drank drunk | drive drove driven | eat ate eaten | fall fell fallen | feel felt
fight fought | find found | fly flew flown | forget forgot forgotten | get got gotten | give gave given
grow grew grown | hear heard | hold held | keep kept | know knew known | lead led | leave left | lend lent
lose lost | make made | mean meant | meet met | pay paid | ride rode ridden | ring rang rung | rise rose risen
run ran | say said | see saw seen | sell sold | send sent | shake shook shaken | shoot shot | sing sang sung
sit sat | sleep slept | speak spoke spoken | spend spent | stand stood | steal stole stolen | swim swam swum
take took taken | teach taught | tear tore torn | tell told | think thought | throw threw thrown
understand understood | wake woke woken | wear wore worn | win won | write wrote written | lie lay lain
child children | man men | woman women | person people | foot feet | tooth teeth | mouse mice
good better best | bad worse worst | far further farther furthest""".split("|")
             for base, *forms in [line.split()] for form in [base, *forms]}


def _bare(words):
    return [w for w in (re.sub(r"[^\w']", "", w.lower()) for w in words) if w]


def _stem(w):
    if w in IRREGULAR:
        return IRREGULAR[w]
    for suffix in ("ies", "ing", "ed", "es", "s", "er", "est", "ly"):
        if w.endswith(suffix) and len(w) - len(suffix) >= 3:
            w = w[:-len(suffix)]
            break
    return w[:-1] if len(w) > 3 and w[-1] == w[-2] else w             # stopped -> stopp -> stop


def _same_word(x, y):
    """go / went, know / knowing, recieve / receive: the same word in another form or spelling."""
    return _stem(x) == _stem(y) or difflib.SequenceMatcher(None, x, y).ratio() >= 0.6


def is_rewrite(src_words, edit):
    """True unless the edit only changes punctuation / case, function words (a -> an, in -> on, + the),
    and the form or spelling of the other words (go -> went, am knowing -> know, recieve -> receive)."""
    i1, i2, repl = edit
    a = [w for w in _bare(src_words[i1:i2]) if w not in FUNCTION_WORDS]
    b = [w for w in _bare(repl) if w not in FUNCTION_WORDS]
    return len(a) != len(b) or not all(_same_word(x, y) for x, y in zip(a, b))


def filter_correction(model, sp, src, hyp, min_gain=MIN_GAIN):
    """Keep only the edits of hyp (a correction of src) that are not rewrites and that the model does not
    clearly reject. Both texts are compared word by word (split on spaces)."""
    words = src.split()
    edits = [e for e in word_edits(src, hyp) if not is_rewrite(words, e)]
    if not edits:
        return src
    scores = score_pairs(model, sp, [(src, src)] + [(src, apply_edits(src, [e])) for e in edits])
    return apply_edits(src, [e for e, s in zip(edits, scores[1:]) if s - scores[0] > min_gain])


def word_edits(src, hyp):
    """Word-level edits that turn src into hyp: (start, end, replacement words) on src.split()."""
    a, b = src.split(), hyp.split()
    return [(i1, i2, b[j1:j2]) for tag, i1, i2, j1, j2 in
            difflib.SequenceMatcher(None, a, b, autojunk=False).get_opcodes() if tag != "equal"]


def apply_edits(src, edits):
    words = src.split()
    for i1, i2, repl in sorted(edits, key=lambda e: -e[0]):   # right to left keeps the indices valid
        words[i1:i2] = repl
    return " ".join(words)


@torch.no_grad()
def score_pairs(model, sp, pairs, batch_size=32):
    """log P(candidate </s> | "Incorrect: <source> Correct:") for each (source, candidate) pair."""
    device = next(model.parameters()).device
    eos = sp.eos_id()
    seqs = []
    for src, cand in pairs:
        p = sp.encode(PROMPT.format(src.strip()), out_type=int)
        seqs.append((p, sp.encode(cand, out_type=int) + [eos]))
    scores = [float("-inf")] * len(seqs)
    order = sorted(range(len(seqs)), key=lambda i: len(seqs[i][0]) + len(seqs[i][1]))
    for start in range(0, len(order), batch_size):
        batch = [i for i in order[start:start + batch_size] if len(seqs[i][0]) + len(seqs[i][1]) <= model.block_size + 1]
        if not batch:
            continue
        T = max(len(seqs[i][0]) + len(seqs[i][1]) for i in batch) - 1
        x = torch.full((len(batch), T), eos, dtype=torch.long)
        y = torch.full((len(batch), T), -100, dtype=torch.long)
        for row, i in enumerate(batch):
            p, a = seqs[i]
            full = torch.tensor(p + a)
            x[row, :len(full) - 1] = full[:-1]
            y[row, len(p) - 1:len(full) - 1] = full[len(p):]
        logp = F.log_softmax(model(x.to(device))[0].float(), dim=-1)
        mask = y != -100
        tok = logp.gather(-1, y.clamp(min=0).to(device).unsqueeze(-1)).squeeze(-1).cpu() * mask
        for row, i in enumerate(batch):
            scores[i] = tok[row].sum().item()
    return scores
