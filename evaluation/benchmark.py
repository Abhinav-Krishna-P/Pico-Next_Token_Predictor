# benchmark.py – score a grammar model on the standard grammar-correction test sets with ERRANT
#
# The 30-sentence test set in evaluation/test_set is good for reading individual corrections, but too
# small to measure progress. This script runs the model on the public benchmarks that research papers
# report, and scores the corrections with ERRANT (span-based correction precision / recall / F0.5):
#
#   bea     BEA-2019 dev (W&I+LOCNESS)    4,384 sentences, 1 reference   (the BEA test answers are hidden)
#   conll   CoNLL-2014 test               1,312 sentences, 2 references
#   jfleg   JFLEG test                      747 sentences, 4 references (fluency rewrites)
#
#   pip install errant && python -m spacy download en_core_web_sm      # once
#   python -m evaluation.benchmark                    # all sets (downloads them into data/benchmarks/ once)
#   python -m evaluation.benchmark --sets bea --limit 300                # quick check on the first 300
#   python -m evaluation.benchmark --model models/other.pt               # another model
#
# Corrections are cached in evaluation/reports/benchmarks/ (one file per model and test set), so re-scoring
# is instant; --rerun regenerates them. Results go to evaluation/reports/benchmarks/<model>.json.
#
# Tokenization: the test sets are word-tokenized ("I do n't know ."), Pico reads normal text. Each sentence
# is detokenized for the model, and in the model's output every word it did not change keeps the original
# tokens; only the changed words are re-tokenized with spaCy. So an unchanged sentence never produces
# spurious edits (a naive re-tokenization adds fake edits to 1.5% of BEA and 10% of CoNLL sentences).
#
# Notes: CoNLL-2014 papers usually report the M2 scorer and JFLEG papers GLEU; the ERRANT numbers here are
# close but not identical to those. The data is licensed for non-commercial research only: it is
# downloaded locally and never committed.

import argparse
import difflib
import json
import os
import re
import subprocess
import sys
import tarfile
import time
import urllib.request

import torch
import torch.nn.functional as F

from pico.grammar import PROMPT, load_grammar_model, resolve_model_path

DATA = "data/benchmarks"
OUT = "evaluation/reports/benchmarks"
SETS = {
    "bea": dict(name="BEA-2019 dev", m2=f"{DATA}/wi+locness/m2/ABCN.dev.gold.bea19.m2",
                url="https://www.cl.cam.ac.uk/research/nl/bea2019st/data/wi+locness_v2.1.bea19.tar.gz"),
    "conll": dict(name="CoNLL-2014 test", m2=f"{DATA}/conll14st-test-data/noalt/official-2014.combined.m2",
                  url="https://www.comp.nus.edu.sg/~nlp/conll14st/conll14st-test-data.tar.gz"),
    "jfleg": dict(name="JFLEG test", m2=f"{DATA}/jfleg_test.m2",
                  url="https://raw.githubusercontent.com/keisks/jfleg/master/test/test.{}"),
}

parser = argparse.ArgumentParser(description="Score a Pico grammar model on BEA-2019 / CoNLL-2014 / JFLEG with ERRANT")
parser.add_argument("--model", default="models/pico_big_grammar_small.pt")
parser.add_argument("--sets", nargs="+", default=list(SETS), choices=list(SETS))
parser.add_argument("--limit", type=int, default=0, help="only the first N sentences of each set (quick checks)")
parser.add_argument("--batch-size", type=int, default=16)
parser.add_argument("--rerun", action="store_true", help="regenerate cached corrections")
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.manual_seed(0)

try:
    import errant
except ImportError:
    sys.exit("ERRANT is needed: pip install errant && python -m spacy download en_core_web_sm")


# ----------------------------
# Test data
# ----------------------------
def download(url, path):
    print(f"  downloading {url}")
    urllib.request.urlretrieve(url, path)


def ensure_data(key, annotator):
    s = SETS[key]
    if os.path.exists(s["m2"]):
        return
    os.makedirs(DATA, exist_ok=True)
    if key in ("bea", "conll"):
        archive = os.path.join(DATA, os.path.basename(s["url"]))
        if not os.path.exists(archive):
            download(s["url"], archive)
        with tarfile.open(archive) as tar:
            tar.extractall(DATA)
        return
    # JFLEG ships plain text (source + 4 references): build the gold M2 file with ERRANT
    parts = {}
    for p in ("src", "ref0", "ref1", "ref2", "ref3"):
        path = f"{DATA}/jfleg_test.{p}"
        if not os.path.exists(path):
            download(s["url"].format(p), path)
        parts[p] = [line.strip() for line in open(path, encoding="utf-8")]
    print("  building the JFLEG gold M2 file with ERRANT")
    with open(s["m2"], "w", encoding="utf-8") as f:
        for i, src in enumerate(parts["src"]):
            orig = annotator.parse(src, False)
            f.write(" ".join(["S"] + [t.text for t in orig]) + "\n")
            for ref_id in range(4):
                ref = parts[f"ref{ref_id}"][i]
                if ref == orig.text.strip():
                    f.write(f"A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||{ref_id}\n")
                else:
                    for e in annotator.annotate(orig, annotator.parse(ref, False)):
                        f.write(e.to_m2(ref_id) + "\n")
            f.write("\n")


def read_m2(path):
    """Blocks of an M2 file: (source tokens, block text)."""
    blocks = open(path, encoding="utf-8").read().strip().split("\n\n")
    return [(b.split("\n")[0][2:].split(), b) for b in blocks]


# ----------------------------
# Detokenize / re-tokenize
# ----------------------------
CLITICS = {"n't", "'s", "'re", "'ve", "'ll", "'d", "'m", "'"}


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
    groups = detok_groups(tokens)
    return " ".join("".join(surface(tokens[i]) for i in g) for g in groups), groups


def retokenize(tokens, groups, hyp, annotator):
    """Tokens of the model output: unchanged words keep the source tokens, changed words go through spaCy."""
    src_words = ["".join(surface(tokens[i]) for i in g) for g in groups]
    hyp_words = hyp.split()
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, src_words, hyp_words, autojunk=False).get_opcodes():
        if tag == "equal":
            out += [tokens[k] for g in groups[i1:i2] for k in g]
        elif j2 > j1:
            out += [t.text for t in annotator.parse(" ".join(hyp_words[j1:j2]), True)]
    return out


def normalize(s):
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.split())


# ----------------------------
# Batched greedy correction
# ----------------------------
@torch.no_grad()
def cached_logits(model, idx, pos0, cache):
    """Run only the new tokens `idx` (at positions pos0..) through the model, reusing the keys and values
    of earlier tokens from `cache` (a KV cache: each step costs one token instead of the whole sequence).
    Same math as Pico.forward; returns the logits of the last position."""
    B, T = idx.shape
    x = model.token_embedding(idx) + model.position_embedding(torch.arange(pos0, pos0 + T))
    for i, blk in enumerate(model.blocks):
        att = blk.sa
        C, H = x.size(-1), att.num_heads
        q, k, v = (t.view(B, T, H, C // H).transpose(1, 2) for t in att.qkv(blk.ln1(x)).split(C, dim=2))
        if cache[i] is not None:
            k, v = torch.cat([cache[i][0], k], 2), torch.cat([cache[i][1], v], 2)
        cache[i] = (k, v)
        out = F.scaled_dot_product_attention(q, k, v, is_causal=pos0 == 0 and T > 1)
        x = x + att.proj(out.transpose(1, 2).reshape(B, T, C))
        x = x + blk.ffwd(blk.ln2(x))
    return model.head(model.ln_f(x[:, -1]))


@torch.no_grad()
def correct_batch(model, sp, sentences):
    """Greedy-decode corrections with a KV cache. Prompts of equal length are batched together, so no
    padding is needed; the output is the same as pico.grammar.correct (checked on BEA sentences)."""
    eos = sp.eos_id()
    prompts = [sp.encode(PROMPT.format(s), out_type=int) for s in sentences]
    order = sorted(range(len(prompts)), key=lambda i: len(prompts[i]))
    results = [None] * len(prompts)
    i = 0
    while i < len(order):
        L = len(prompts[order[i]])
        batch = [order[i]]
        while i + len(batch) < len(order) and len(batch) < args.batch_size and len(prompts[order[i + len(batch)]]) == L:
            batch.append(order[i + len(batch)])
        i += len(batch)
        max_new = min(model.block_size - L, int(1.5 * L) + 16)
        if max_new <= 0:
            for b in batch:
                results[b] = ""
            continue
        idx = torch.tensor([prompts[b] for b in batch], dtype=torch.long)
        done = torch.zeros(len(batch), dtype=torch.bool)
        cache = [None] * len(model.blocks)
        logits = cached_logits(model, idx, 0, cache)
        for _ in range(max_new):
            nxt = logits.argmax(-1)
            idx = torch.cat([idx, nxt[:, None]], 1)
            if idx.size(1) < model.block_size:
                logits = cached_logits(model, nxt[:, None], idx.size(1) - 1, cache)
            done |= nxt == eos
            if done.all():
                break
        for row, b in zip(idx[:, L:].tolist(), batch):
            results[b] = sp.decode(row[:row.index(eos)] if eos in row else row)
    return results


# ----------------------------
# Scoring (official ERRANT scorer)
# ----------------------------
def errant_compare(hyp_m2, ref_m2):
    env = dict(os.environ, PYTHONUTF8="1")
    cmd = [sys.executable, "-c", "from errant.commands.compare_m2 import main; main()",
           "-hyp", hyp_m2, "-ref", ref_m2, "-cat", "2"]
    out = subprocess.run(cmd, capture_output=True, text=True, env=env, encoding="utf-8").stdout
    lines = out.strip().split("\n")
    head = lines.index(next(l for l in lines if l.startswith("TP\tFP")))
    tp, fp, fn, p, r, f = lines[head + 1].split("\t")
    cats = {}
    for line in lines:
        parts = line.split()
        if len(parts) == 7 and parts[1].isdigit() and parts[0] != "Category":
            cats[parts[0]] = dict(tp=int(parts[1]), fp=int(parts[2]), fn=int(parts[3]),
                                  p=float(parts[4]), r=float(parts[5]), f05=float(parts[6]))
    return dict(tp=int(tp), fp=int(fp), fn=int(fn), precision=float(p), recall=float(r), f05=float(f)), cats


def run_set(key, model, sp, annotator, model_name):
    s = SETS[key]
    ensure_data(key, annotator)
    blocks = read_m2(s["m2"])
    if args.limit:
        blocks = blocks[:args.limit]
    n = len(blocks)
    tag = f"{model_name}.{key}" + (f".first{args.limit}" if args.limit else "")
    cache = f"{OUT}/{tag}.txt"

    sources = [detok(tokens) for tokens, _ in blocks]
    if os.path.exists(cache) and not args.rerun:
        hyps = open(cache, encoding="utf-8").read().split("\n")[:n]
        print(f"  {n} corrections from {cache}")
        seconds = None
    else:
        t0, hyps = time.perf_counter(), []
        step = 256
        for start in range(0, n, step):
            hyps += correct_batch(model, sp, [text for text, _ in sources[start:start + step]])
            done = min(start + step, n)
            el = time.perf_counter() - t0
            print(f"  {done}/{n} sentences | {el:.0f} s | ~{el / done * (n - done):.0f} s left", flush=True)
        seconds = time.perf_counter() - t0
        with open(cache, "w", encoding="utf-8") as f:
            f.write("\n".join(h.replace("\n", " ") for h in hyps))

    hyp_m2, ref_m2 = f"{OUT}/{tag}.hyp.m2", f"{OUT}/{tag}.ref.m2"
    changed = 0
    with open(hyp_m2, "w", encoding="utf-8") as fh, open(ref_m2, "w", encoding="utf-8") as fr:
        for (tokens, block), (text, groups), hyp in zip(blocks, sources, hyps):
            hyp = normalize(hyp)
            orig = annotator.parse(" ".join(tokens), False)
            fh.write(" ".join(["S"] + tokens) + "\n")
            if hyp == text:
                fh.write("A -1 -1|||noop|||-NONE-|||REQUIRED|||-NONE-|||0\n")
            else:
                changed += 1
                cor = annotator.parse(" ".join(retokenize(tokens, groups, hyp, annotator)), False)
                for e in annotator.annotate(orig, cor):
                    fh.write(e.to_m2(0) + "\n")
            fh.write("\n")
            fr.write(block + "\n\n")
    total, cats = errant_compare(hyp_m2, ref_m2)
    if key == "conll":
        cats = {}   # CoNLL gold uses NUCLE error types (ArtOrDet, Prep, ...), not ERRANT's: no per-type breakdown
    return dict(name=s["name"], sentences=n, changed=changed, seconds=seconds, **total, categories=cats)


# ----------------------------
# Main
# ----------------------------
path = resolve_model_path(args.model)
model_name = os.path.splitext(os.path.basename(path))[0]
print(f"Model {path}")
model, sp, ckpt = load_grammar_model(path, "cpu")
print("Loading ERRANT (spaCy en_core_web_sm)")
annotator = errant.load("en")
os.makedirs(OUT, exist_ok=True)

results = {}
for key in args.sets:
    print(f"\n{SETS[key]['name']}")
    r = results[key] = run_set(key, model, sp, annotator, model_name)
    print(f"  P {r['precision']:.4f} | R {r['recall']:.4f} | F0.5 {r['f05']:.4f} "
          f"(TP {r['tp']}, FP {r['fp']}, FN {r['fn']}; changed {r['changed']}/{r['sentences']} sentences)")

summary_path = f"{OUT}/{model_name}" + (f".first{args.limit}" if args.limit else "") + ".json"
saved = json.load(open(summary_path, encoding="utf-8"))["results"] if os.path.exists(summary_path) else {}
saved.update(results)   # running one set keeps the scores of the others
with open(summary_path, "w", encoding="utf-8") as f:
    json.dump(dict(model=path, params=sum(p.numel() for p in model.parameters()), limit=args.limit or None,
                   results={k: saved[k] for k in SETS if k in saved}), f, indent=1)

print(f"\n{'Test set':18s} {'Sents':>6s} {'P':>7s} {'R':>7s} {'F0.5':>7s}")
for r in results.values():
    print(f"{r['name']:18s} {r['sentences']:6d} {r['precision']:7.4f} {r['recall']:7.4f} {r['f05']:7.4f}")
for key, r in results.items():
    if not r["categories"]:
        continue
    top =sorted(r["categories"].items(), key=lambda kv: -(kv[1]["tp"] + kv[1]["fn"]))[:8]
    print(f"\n{r['name']} – most frequent error types (gold edits):")
    for cat, c in top:
        print(f"  {cat:14s} gold {c['tp'] + c['fn']:5d} | fixed {c['tp']:4d} | wrong edits {c['fp']:4d} | F0.5 {c['f05']:.2f}")
print(f"\nResults: {summary_path}")
