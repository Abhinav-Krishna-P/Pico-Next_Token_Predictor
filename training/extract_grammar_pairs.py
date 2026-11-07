# extract_grammar_pairs.py – turn the English Practical Grammar dataset into incorrect -> correct pairs
#
# The dataset (github.com/Abhinav-Krishna-P/English-practical-grammer-Dataset) is a chat-style tutor
# dataset. Pico only does sentence correction, so the exercises are converted into the same
# {"input": wrong, "output": correct} format as train.jsonl:
#   - "Correct the following sentence" / "Correct the preposition / tense / adjective …" -> as given
#   - "Choose the correct verb: … ___ … (a / b)"           -> the blank filled with the wrong option
#   - "Fill in the blank with a suitable preposition: …" -> the blank filled with a wrong preposition
#   - "Supply a, an, or the …"                           -> the sentence without its article
#   - "Use the correct tense of the verb given in brackets: …" -> the bare verb left in place
#   - "Punctuate the following sentence correctly"       -> the unpunctuated sentence
# The full conversations (all rows, including explanations) are trained separately as Question -> Answer
# by finetune_grammar.py, so no data is lost; this script only adds the extra correction pairs.
#
# Test sentences from "model  testing/" that also appear in the dataset are listed in
# <out>_test_overlap.json, so evaluation/evaluate.py can report scores with and without them.
# Use --exclude-test to leave those pairs out of training instead.
#
#   git clone https://github.com/Abhinav-Krishna-P/English-practical-grammer-Dataset data/english-practical-grammar
#   python -m training.extract_grammar_pairs

import argparse
import collections
import difflib
import glob
import json
import os
import random
import re
import sys

parser = argparse.ArgumentParser(description="Extract grammar-correction pairs from the English Practical Grammar dataset")
parser.add_argument("--src", default="data/english-practical-grammar", help="cloned dataset repository")
parser.add_argument("--out", default="data/grammar_book.jsonl")
parser.add_argument("--variants", type=int, default=3, help="wrong versions per fill-in-the-blank exercise")
parser.add_argument("--test", nargs="*", default=["model  testing/qns.txt", "model  testing/ans.txt"],
                    help="test-set files, checked for overlap with the dataset")
parser.add_argument("--exclude-test", action="store_true", help="drop pairs that are near-copies of test sentences")
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
random.seed(1337)

PREPOSITIONS = ["in", "on", "at", "to", "for", "of", "with", "from", "by", "about", "into", "over", "against", "upon"]


def clean(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    s = s.replace("*", "")                         # markdown emphasis (blanks "___" are kept)
    s = re.sub(r"\s*\((?:or|also)\b[^)]*\)", "", s)  # "(or …)" alternatives
    return " ".join(s.split()).strip()


def first_bold_sentence(answer):
    """The first **bold** span that looks like a sentence (skips labels such as **Incorrect.**)."""
    for span in re.findall(r"\*\*(.+?)\*\*", answer):
        s = clean(span)
        if len(s.split()) >= 3 and not re.match(r"^(in)?correct\b", s, re.I):
            return s
    return None


def italic_sentence(answer):
    """Sentences given as *…* with the answer word in **bold** inside (preposition exercises)."""
    m = re.search(r"\*((?:[^*]|\*\*[^*]+\*\*)+?)\*(?!\*)", answer.split("\n")[0].split("—", 1)[-1])
    return clean(m.group(1)) if m else None


def exercise_text(user):
    """The sentence an exercise is about: quoted, or the text after the instruction line."""
    m = re.search(r'"([^"]{8,})"', user.replace("“", '"').replace("”", '"'))
    if m:
        return m.group(1)
    parts = user.split("\n", 1)
    if len(parts) > 1:
        return parts[1]
    return user.split(":", 1)[1] if ":" in user else None   # "Choose the correct verb: … ___ … (a / b)"


def is_sentence(s):
    return s and 3 <= len(s.split()) <= 40 and "\n" not in s


pairs, counts = [], collections.Counter()


def add(kind, wrong, right):
    wrong, right = clean(wrong), clean(right)
    if is_sentence(wrong) and is_sentence(right):
        pairs.append({"input": wrong, "output": right, "kind": kind})
        counts[kind] += 1


rows = [json.loads(line) for f in sorted(glob.glob(os.path.join(args.src, "data", "*.jsonl")))
        for line in open(f, encoding="utf-8") if line.strip()]
if not rows:
    sys.exit(f"no dataset found in {args.src}: clone it first (see the top of this file)")

for r in rows:
    user, answer = r["messages"][1]["content"], r["messages"][2]["content"]
    first = user.split("\n")[0]
    text = exercise_text(user)
    if not text:
        continue
    text = clean(text)

    if re.match(r"^(Correct (the|this|any)|Is this sentence correct|Rewrite .*correct)", first, re.I) or \
            re.search(r"\bCorrect it\b", first):
        right = first_bold_sentence(answer)
        if right:
            add("correction", text, right)
        elif re.match(r"^\**correct\b", answer.strip(), re.I):
            add("already correct", text, text)

    elif re.match(r"^(Choose the correct verb|Fill in the blank, choosing)", first):
        opts = re.search(r"\(([^()]+/[^()]+)\)\s*\.?$", text)
        right = first_bold_sentence(answer)
        if opts and right and "___" in text:
            stem = text[:opts.start()].strip()
            options = [o.strip() for o in opts.group(1).split("/")]
            for o in options:
                wrong = re.sub(r"_{2,}", o, stem, count=1)
                if clean(wrong) != right:
                    add("choose the verb", wrong, right)

    elif re.match(r"^(Fill in the blank with a suitable preposition|Use a suitable preposition|Supply a suitable preposition)", first):
        right = italic_sentence(answer) or first_bold_sentence(answer)
        prep = re.match(r"\*\*([a-z ]+)\*\*", answer.strip())
        if right and prep and "---" in text:
            prep = prep.group(1).strip()
            for wrong_prep in random.sample([p for p in PREPOSITIONS if p != prep], args.variants):
                add("preposition", text.replace("---", wrong_prep, 1), right)

    elif first.startswith("Supply a, an, or the"):
        right = first_bold_sentence(answer)
        if right and "---" in text:
            add("article", text.replace("---", ""), right)
            add("article", text.replace("---", random.choice(["a", "an", "the"])), right)

    elif re.match(r"^Use the correct tense of the verb", first):
        right = first_bold_sentence(answer)
        if right and re.search(r"\(\w+\)", text):
            add("tense", re.sub(r"\((\w+)\)", r"\1", text), right)

    elif first.startswith("Punctuate the following"):
        right = first_bold_sentence(answer)
        if right:
            add("punctuation", text, right)


# ----------------------------
# Keep the benchmark fair: drop pairs that are (near) copies of a test sentence
# ----------------------------
def norm(s):
    return re.sub(r"[^a-z ]", "", s.lower()).strip()


def numbered(path):
    return {int(m.group(1)): m.group(2) for m in (re.match(r"^\s*(\d+)\.\s*(.*\S)", line)
            for line in open(path, encoding="utf-8")) if m} if os.path.exists(path) else {}


def similar(a, b):
    return difflib.SequenceMatcher(None, norm(a), norm(b)).ratio() > 0.75


test_q, test_a = (numbered(p) for p in (args.test + ["", ""])[:2])
test = list(test_q.values()) + list(test_a.values())

# which test sentences appear anywhere in the dataset (conversations included)?
texts = [m["content"] for r in rows for m in r["messages"][1:]]
candidates = [clean(t) for text in texts for t in re.split(r'(?<=[.!?"])\s+|\n|: ', text) if 3 <= len(t.split()) <= 25]
overlap = sorted(i for i, q in test_q.items()
                 if any(similar(q, c) or similar(test_a.get(i, ""), c) for c in candidates))

dropped = [p for p in pairs if args.exclude_test and any(similar(s, t) for s in (p["input"], p["output"]) for t in test)]
seen, unique = set(), []
for p in pairs:
    if p not in dropped and (p["input"], p["output"]) not in seen:
        seen.add((p["input"], p["output"]))
        unique.append(p)

os.makedirs(os.path.dirname(os.path.abspath(args.out)), exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    for p in unique:
        f.write(json.dumps({"input": p["input"], "output": p["output"]}) + "\n")
with open(os.path.splitext(args.out)[0] + "_test_overlap.json", "w", encoding="utf-8") as f:
    json.dump({"test_ids_in_training_data": overlap, "excluded_from_pairs": args.exclude_test}, f, indent=1)

print(f"{len(rows):,} dataset rows -> {len(unique):,} correction pairs written to {args.out}")
for kind, n in counts.most_common():
    print(f"  {kind:16s} {n}")
if args.exclude_test:
    print(f"  dropped {len(dropped)} pair(s) similar to the test set")
print(f"Test sentences that also appear in the dataset: {overlap or 'none'}")
for i in overlap:
    print(f"  {i:2d}. {test_q[i]}")
