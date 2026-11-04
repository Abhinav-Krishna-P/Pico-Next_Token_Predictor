# make_synthetic.py – generate short grammar-correction pairs with common learner mistakes
#
# Clean sentences (from TinyStories + FineWeb-Edu, or local text files) are broken on purpose:
#     "He doesn't like coffee."  ->  "He dont like coffee."
#     "I ate an apple yesterday." -> "I eat a apple yesterday."
# and saved as {"input": broken, "output": clean} lines, the same format as train.jsonl.
#
#   python make_synthetic.py --out data/synthetic.jsonl                                  # stream from Hugging Face
#   python make_synthetic.py --out data/synthetic.jsonl --text-files data/stories.txt --pairs 20000   # local files

import argparse
import json
import os
import random
import re
import sys
import traceback

parser = argparse.ArgumentParser(description="Generate synthetic grammar-correction pairs")
parser.add_argument("--out", default="data/synthetic.jsonl")
parser.add_argument("--pairs", type=int, default=400_000)
parser.add_argument("--web-share", type=float, default=0.5, help="fraction of sentences from FineWeb-Edu")
parser.add_argument("--clean-share", type=float, default=0.1,
                    help="fraction of already-correct pairs (teaches the model to leave good sentences alone)")
parser.add_argument("--text-files", nargs="*", help="use local text files instead of downloading")
parser.add_argument("--words", default="words_alpha.txt", help="English word list (for -ed / -ing stems)")
args = parser.parse_args()
random.seed(1337)


# Streaming downloads leave background threads running that can stop Python from exiting
# (the process then hangs forever, e.g. after an error). Always exit hard instead.
def hard_exit(code):
    sys.stdout.flush()
    sys.stderr.flush()
    os._exit(code)


sys.excepthook = lambda *exc: (traceback.print_exception(*exc), hard_exit(1))

WORDS = set()
if os.path.exists(args.words):
    WORDS = {w.strip() for w in open(args.words, encoding="utf-8")}


# ----------------------------
# Source sentences
# ----------------------------
def sentences(text):
    for s in re.split(r"(?<=[.!?])\s+", " ".join(text.split())):
        n = len(s.split())
        if 4 <= n <= 30 and s[0].isupper() and s[-1] in ".!?" and s.isascii() and "http" not in s:
            yield s


def stream(dataset_args, limit):
    from datasets import load_dataset
    found = 0
    for row in load_dataset(*dataset_args, split="train", streaming=True):
        for s in sentences(row["text"]):
            yield s
            found += 1
            if found >= limit:
                return


def collect():
    if args.text_files:
        out = []
        for path in args.text_files:
            for doc in open(path, encoding="utf-8").read().split("<|endoftext|>"):
                out.extend(sentences(doc))
        random.shuffle(out)
        return out[:args.pairs]
    n_web = int(args.pairs * args.web_share)
    print("Downloading TinyStories sentences...")
    out = list(stream(("roneneldan/TinyStories",), args.pairs - n_web))
    print("Downloading FineWeb-Edu sentences...")
    out += stream(("HuggingFaceFW/fineweb-edu", "sample-10BT"), n_web)
    random.shuffle(out)
    return out


# ----------------------------
# Corruptions: each takes a list of words and returns a changed list, or None if it doesn't apply
# ----------------------------
TOKEN = re.compile(r"^(\W*)([A-Za-z']+?)(\W*)$")
VOWEL_SOUND = re.compile(r"^(?:[aeio]|u(?!ni|se|s[eu]|r[eio]))|^h(?:our|onest|onou?r)", re.I)


def parts(w):
    m = TOKEN.match(w)
    return m.groups() if m else ("", "", "")


def core(w):
    return parts(w)[1].lower()


def replaced(ws, i, new):
    pre, old, post = parts(ws[i])
    if old[:1].isupper():
        new = new[:1].upper() + new[1:]
    return ws[:i] + [pre + new + post] + ws[i + 1:]


def pick(ws, test):
    hits = [i for i in range(len(ws)) if test(i)]
    return random.choice(hits) if hits else None


def swap_from(table):
    def f(ws):
        i = pick(ws, lambda i: core(ws[i]) in table)
        if i is None:
            return None
        new = table[core(ws[i])]
        return replaced(ws, i, random.choice(new) if isinstance(new, list) else new)
    return f


AGREEMENT = {"was": "were", "were": "was", "is": "are", "are": "is", "has": "have", "have": "has",
             "does": "do", "do": "does", "doesn't": "don't", "don't": "doesn't", "isn't": "aren't",
             "aren't": "isn't", "wasn't": "weren't", "weren't": "wasn't", "am": ["is", "are"]}
PAST = {"went": ["go", "goes"], "saw": "see", "ate": "eat", "came": "come", "took": "take", "gave": "give",
        "made": "make", "said": "say", "found": "find", "ran": "run", "had": "have", "got": "get",
        "knew": "know", "thought": "think", "told": "tell", "felt": "feel", "bought": "buy", "wrote": "write",
        "began": "begin", "sat": "sit", "stood": "stand", "left": "leave", "brought": "bring", "kept": "keep",
        "met": "meet", "heard": "hear", "flew": "fly", "swam": "swim", "drank": "drink", "sang": "sing",
        "fell": "fall", "won": "win", "lost": "lose", "built": "build", "taught": "teach", "caught": "catch",
        "spoke": "speak", "broke": "break", "drove": "drive", "rode": "ride", "became": "become",
        "seen": "saw", "done": "did", "gone": "went", "eaten": "ate", "written": "wrote", "taken": "took"}
CONFUSED = {"their": ["there", "they're"], "there": "their", "they're": "their", "your": "you're",
            "you're": "your", "its": "it's", "it's": "its", "then": "than", "than": "then", "too": "to",
            "to": "too", "lose": "loose", "affect": "effect", "effect": "affect", "who's": "whose",
            "whose": "who's", "fewer": "less", "much": "many", "many": "much", "good": "well", "well": "good"}
PREPOSITIONS = {"in": ["on", "at"], "on": ["in", "at"], "at": ["in", "on"], "for": "to", "of": "for",
                "with": "to", "from": "of", "about": "of", "into": "in", "during": "in"}
COMPARATIVE = {"better": "more good", "worse": "more bad", "bigger": "more big", "best": "most good",
               "happier": "more happy", "faster": "more fast", "older": "more old", "easier": "more easy"}
PLURAL_HINT = {"two", "three", "four", "five", "many", "several", "few", "some", "these", "those", "all"}
BE = {"am", "is", "are", "was", "were", "be", "been", "i'm", "he's", "she's", "it's", "we're", "they're",
      "you're"}
SUBJ3 = {"he", "she", "it"}
NOT_VERB_S = {"was", "is", "has", "does", "always", "also", "this", "thus", "as", "us", "yes", "its", "his"}


def article_an(ws):
    # "an apple" -> "a apple", or "a dog" -> "an dog"
    i = pick(ws, lambda i: i + 1 < len(ws) and core(ws[i]) in ("a", "an") and core(ws[i + 1]))
    if i is None:
        return None
    return replaced(ws, i, "an" if core(ws[i]) == "a" else "a")


def drop_article(ws):
    i = pick(ws, lambda i: core(ws[i]) in ("a", "an", "the") and not parts(ws[i])[0] and not parts(ws[i])[2])
    if i is None:
        return None
    out = ws[:i] + ws[i + 1:]
    if i == 0 and out:
        out = replaced(out, 0, parts(out[0])[1])
    return out


def drop_apostrophe(ws):
    # don't -> dont, I'm -> Im, it's -> its
    i = pick(ws, lambda i: "'" in core(ws[i]))
    if i is None:
        return None
    return replaced(ws, i, parts(ws[i])[1].replace("'", ""))


def third_person(ws):
    # "she likes" -> "she like",  "he goes" -> "he go",  "they play" -> "they plays"
    def test(i):
        if i == 0 or not core(ws[i]) or parts(ws[i - 1])[2]:
            return False
        prev, w = core(ws[i - 1]), core(ws[i])
        return (prev in SUBJ3 and w.endswith("s") and len(w) > 3 and w not in NOT_VERB_S) or \
               (prev in ("i", "you", "we", "they") and w in WORDS and w + "s" in WORDS and len(w) > 2
                and not w.endswith(("s", "ed", "ing", "ly")))
    i = pick(ws, test)
    if i is None:
        return None
    w = core(ws[i])
    if core(ws[i - 1]) not in SUBJ3:
        return replaced(ws, i, w + "s")
    if w.endswith("ies"):
        base = w[:-3] + "y"
    elif w.endswith(("ches", "shes", "sses", "xes", "oes")):
        base = w[:-2]
    else:
        base = w[:-1]
    return replaced(ws, i, base)


def past_tense(ws):
    # "went" -> "go", "played" -> "play"
    def test(i):
        w = core(ws[i])
        return w in PAST or (w.endswith("ed") and len(w) > 4 and w in WORDS)
    i = pick(ws, test)
    if i is None:
        return None
    w = core(ws[i])
    if w in PAST:
        new = PAST[w]
        return replaced(ws, i, random.choice(new) if isinstance(new, list) else new)
    stem = w[:-3] + "y" if w.endswith("ied") else w[:-2]
    if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
        stem = stem[:-1]                         # stopped -> stop
    elif stem + "e" in WORDS and stem not in WORDS:
        stem += "e"                              # liked -> like
    return replaced(ws, i, stem)


def drop_ing(ws):
    # "I am working" -> "I am work"
    def test(i):
        w = core(ws[i])
        return i > 0 and core(ws[i - 1]) in BE and w.endswith("ing") and len(w) > 5
    i = pick(ws, test)
    if i is None:
        return None
    stem = core(ws[i])[:-3]
    if len(stem) > 2 and stem[-1] == stem[-2] and stem[-1] not in "lsz":
        stem = stem[:-1]                         # running -> run
    elif stem + "e" in WORDS and stem not in WORDS:
        stem += "e"                              # making -> make
    return replaced(ws, i, stem)


def wrong_ing(ws):
    # "I want to go" -> "I want to going",  "can swim" -> "can swimming"
    def test(i):
        return i > 0 and core(ws[i - 1]) in ("to", "can", "will", "should", "must", "could", "would") and \
            core(ws[i]) in WORDS and core(ws[i]) + "ing" in WORDS
    i = pick(ws, test)
    return None if i is None else replaced(ws, i, core(ws[i]) + "ing")


def drop_be(ws):
    # "I am happy" -> "I happy"
    i = pick(ws, lambda i: 0 < i < len(ws) - 1 and core(ws[i]) in ("am", "is", "are") and not parts(ws[i])[2])
    return None if i is None else ws[:i] + ws[i + 1:]


def plural(ws):
    # "two cats" -> "two cat"
    def test(i):
        w = core(ws[i])
        return i > 0 and core(ws[i - 1]) in PLURAL_HINT and w.endswith("s") and len(w) > 3 and w[:-1] in WORDS
    i = pick(ws, test)
    return None if i is None else replaced(ws, i, core(ws[i])[:-1])


def lowercase(ws):
    # "I" -> "i", or a lowercase first letter
    i = pick(ws, lambda i: core(ws[i]) in ("i", "i'm", "i've", "i'll", "i'd") and parts(ws[i])[1][0] == "I")
    if i is not None:
        return ws[:i] + [ws[i].replace("I", "i", 1)] + ws[i + 1:]
    if ws[0][:1].isupper():
        return [ws[0][0].lower() + ws[0][1:]] + ws[1:]
    return None


def drop_punct(ws):
    if len(ws[-1]) > 1 and ws[-1][-1] in ".!?":
        return ws[:-1] + [ws[-1][:-1]]
    return None


def duplicate(ws):
    i = random.randrange(len(ws))
    return ws[:i + 1] + [parts(ws[i])[1].lower() or ws[i]] + ws[i + 1:] if core(ws[i]) else None


CORRUPTIONS = [  # (function, weight)
    (article_an, 3), (drop_article, 3), (drop_apostrophe, 3), (swap_from(AGREEMENT), 4), (third_person, 4),
    (past_tense, 4), (drop_ing, 2), (wrong_ing, 2), (drop_be, 1), (plural, 2), (swap_from(CONFUSED), 2),
    (swap_from(PREPOSITIONS), 2), (swap_from(COMPARATIVE), 1), (lowercase, 0.5), (drop_punct, 0.7), (duplicate, 0.7),
]


def corrupt(sentence):
    ws = sentence.split()
    n = random.choices([1, 2, 3], weights=[0.55, 0.33, 0.12])[0]
    funcs, weights = zip(*CORRUPTIONS)
    applied = 0
    for _ in range(12):
        new = random.choices(funcs, weights)[0](ws)
        new = [w for w in new or [] if w]
        if len(new) >= 2 and new != ws:
            ws, applied = new, applied + 1
            if applied == n:
                break
    return " ".join(ws) if applied else None


# ----------------------------
# Generate
# ----------------------------
source = collect()
print(f"{len(source):,} clean sentences")
os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
written = 0
with open(args.out + ".tmp", "w", encoding="utf-8") as f:   # a crash never leaves a half-written file
    for s in source:
        bad = s if random.random() < args.clean_share else corrupt(s)
        if bad is None:
            continue
        f.write(json.dumps({"input": bad, "output": s}) + "\n")
        written += 1
os.replace(args.out + ".tmp", args.out)
print(f"Wrote {written:,} pairs to {args.out}")
hard_exit(0)
