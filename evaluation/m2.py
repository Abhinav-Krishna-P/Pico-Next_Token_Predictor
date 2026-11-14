# m2.py – read M2 grammar-correction files and turn tokenized sentences back into normal text
#
# M2 is the format of the BEA-2019, FCE and CoNLL-2014 data: a tokenized source sentence ("S ...") followed by
# its edits ("A start end|||type|||correction|||REQUIRED|||-NONE-|||annotator"). Used by evaluation/benchmark.py
# and training/prepare_learner_data.py.

import re

CLITICS = {"n't", "'s", "'re", "'ve", "'ll", "'d", "'m", "'"}


def read_m2(path):
    """Blocks of an M2 file: (source tokens, block text)."""
    blocks = open(path, encoding="utf-8").read().strip().split("\n\n")
    return [(b.split("\n")[0][2:].split(), b) for b in blocks]


def corrected_tokens(tokens, block, annotator=0):
    """Apply one annotator's edits to the source tokens. Returns None if the block has an UNK edit
    (an error the annotator marked but did not correct), so it cannot be used as a training target."""
    edits = []
    for line in block.split("\n")[1:]:
        span, etype, cor, *_, ann = line[2:].split("|||")
        if int(ann) != annotator or etype == "noop":
            continue
        if etype == "UNK":
            return None
        start, end = map(int, span.split())
        edits.append((start, end, cor.split()))
    out = list(tokens)
    for start, end, cor in sorted(edits, key=lambda e: -e[0]):   # right to left keeps the offsets valid
        out[start:end] = cor
    return out


def detok_groups(tokens):
    """Glue tokens into whitespace-separated words the way normal text is written.
    Returns a list of groups (lists of token indices); each group is one word of the text."""
    groups, quote_open, glue_next = [], False, False
    for i, t in enumerate(tokens):
        is_quote = t in ('"', "``", "''")
        closing = is_quote and (t == "''" or (t == '"' and quote_open))
        attach = groups and (glue_next or t.lower() in CLITICS or closing or re.fullmatch(r"[,.!?;:%)\]}]+", t))
        if attach:
            groups[-1].append(i)
        else:
            groups.append([i])
        glue_next = t in ("(", "[", "{", "$") or (is_quote and not closing)
        if is_quote:
            quote_open = not closing
    return groups


def surface(tok):
    return '"' if tok in ("``", "''") else tok


def detok(tokens):
    """Normal text for a tokenized sentence, and the token groups behind each of its words."""
    groups = detok_groups(tokens)
    return " ".join("".join(surface(tokens[i]) for i in g) for g in groups), groups
