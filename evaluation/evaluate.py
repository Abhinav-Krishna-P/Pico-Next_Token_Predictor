# evaluate.py – benchmark a grammar model on a test set and write a one-page HTML report
#
# For every numbered sentence in the questions file, the model's correction is compared with the
# answer key. Reported: model size, tokens, latency, loss / perplexity and correction quality
# (exact match, word error rate before vs after, edit precision / recall / F0.5).
#
#   python -m evaluation.evaluate                                  # big model vs small baseline
#   python -m evaluation.evaluate --model models/pico_grammar.pt --baseline none

import argparse
import difflib
import html
import json
import math
import os
import platform
import re
import statistics
import sys
import time
from datetime import datetime

import torch

from pico.grammar import PROMPT, correct, load_grammar_model, resolve_model_path

parser = argparse.ArgumentParser(description="Evaluate a Pico grammar model and write an HTML report")
parser.add_argument("--model", default="models/pico_big_grammar_small.pt")
parser.add_argument("--baseline", default="models/pico_grammar.pt", help="model to compare against, or 'none'")
parser.add_argument("--pretrain", default="models/pico_big.pt", help="pre-training checkpoint (for training stats)")
parser.add_argument("--questions", default="model  testing/qns.txt")
parser.add_argument("--answers", default="model  testing/ans.txt")
parser.add_argument("--out", default="evaluation/reports/report.html")
parser.add_argument("--runs", type=int, default=3, help="timed runs per sentence (median is reported)")
parser.add_argument("--overlap", default=None,
                    help="for models trained on the grammar dataset: data/grammar_book_test_overlap.json "
                         "(test sentences that also appear in the training data are marked and scored separately)")
args = parser.parse_args()
sys.stdout.reconfigure(encoding="utf-8", errors="replace")
torch.manual_seed(0)
device = "cpu"   # report laptop latency


# ----------------------------
# Test set
# ----------------------------
def normalize(s):
    # curly quotes from word processors -> plain ASCII, single spaces
    s = s.replace("’", "'").replace("‘", "'").replace("“", '"').replace("”", '"')
    return " ".join(s.split())


def read_numbered(path):
    items = {}
    for line in open(path, encoding="utf-8"):
        m = re.match(r"^\s*(\d+)\.\s+(.*\S)\s*$", line)
        if m:
            items[int(m.group(1))] = normalize(m.group(2))
    return items


questions, answers = read_numbered(args.questions), read_numbered(args.answers)
ids = sorted(set(questions) & set(answers))
if not ids:
    sys.exit(f"no numbered sentences found in {args.questions} / {args.answers}")
print(f"{len(ids)} test sentences")


# ----------------------------
# Metrics
# ----------------------------
def words(s):
    return re.findall(r"[\w']+|[^\w\s]", s)


def edit_distance(a, b):
    row = list(range(len(b) + 1))
    for i, x in enumerate(a, 1):
        prev, row[0] = row[0], i
        for j, y in enumerate(b, 1):
            prev, row[j] = row[j], min(row[j] + 1, row[j - 1] + 1, prev + (x != y))
    return row[-1]


def edits(src, tgt):
    """Word-level edits that turn src into tgt, as comparable (start, end, replacement) spans."""
    ops = difflib.SequenceMatcher(None, src, tgt, autojunk=False).get_opcodes()
    return {(i1, i2, tuple(tgt[j1:j2])) for tag, i1, i2, j1, j2 in ops if tag != "equal"}


def loose(s):
    return re.sub(r"[.!?]+$", "", s.strip()).lower()


@torch.no_grad()
def answer_loss(model, sp, src, answer):
    """Cross-entropy of `answer` as the correction of `src` (only the answer tokens count)."""
    p = sp.encode(PROMPT.format(src), out_type=int)
    a = sp.encode(answer, out_type=int) + [sp.eos_id()]
    seq = torch.tensor([p + a])
    x, y = seq[:, :-1], seq[:, 1:].clone()
    y[:, :len(p) - 1] = -100
    return model(x, y)[1].item()


def evaluate(path):
    t0 = time.perf_counter()
    model, sp, ckpt = load_grammar_model(path, device)
    load_s = time.perf_counter() - t0
    for s in ("warm up", "the model"):
        correct(model, sp, s)

    rows = []
    for i in ids:
        src, ref = questions[i], answers[i]
        times = []
        for _ in range(args.runs):
            t = time.perf_counter()
            pred, n_prompt, n_new = correct(model, sp, src)
            times.append(time.perf_counter() - t)
        pred = normalize(pred)
        ws, wr, wp = words(src), words(ref), words(pred)
        gold, hyp = edits(ws, wr), edits(ws, wp)
        err_before, err_after = edit_distance(ws, wr), edit_distance(wp, wr)
        loss_ref, loss_src = answer_loss(model, sp, src, ref), answer_loss(model, sp, src, src)
        if pred == ref:
            outcome = "correct"
        elif loose(pred) == loose(ref):
            outcome = "minor"          # only capitalization / final punctuation differ
        elif wp == ws:
            outcome = "unchanged"
        elif err_after < err_before:
            outcome = "improved"
        elif err_after > err_before:
            outcome = "worse"
        else:
            outcome = "nogain"
        latency = statistics.median(times)
        rows.append(dict(id=i, src=src, ref=ref, pred=pred, outcome=outcome, latency_ms=latency * 1000,
                         prompt_tokens=n_prompt, new_tokens=n_new, tok_s=n_new / latency if latency else 0,
                         err_before=err_before, err_after=err_after, ref_words=len(wr),
                         tp=len(gold & hyp), n_gold=len(gold), n_hyp=len(hyp),
                         loss_ref=loss_ref, loss_src=loss_src))
        print(f"  {i:2d}. [{outcome:9s}] {latency * 1000:6.0f} ms  {src}  ->  {pred}", flush=True)

    n_params = sum(p.numel() for p in model.parameters())
    emb = model.token_embedding.weight.numel() + model.position_embedding.weight.numel()
    tp, ng, nh = (sum(r[k] for r in rows) for k in ("tp", "n_gold", "n_hyp"))
    prec, rec = (tp / nh if nh else 0.0), (tp / ng if ng else 0.0)
    f05 = 1.25 * prec * rec / (0.25 * prec + rec) if prec + rec else 0.0
    before, after = sum(r["err_before"] for r in rows), sum(r["err_after"] for r in rows)
    ref_words = sum(r["ref_words"] for r in rows)
    lat = [r["latency_ms"] for r in rows]
    mean_loss = statistics.mean(r["loss_ref"] for r in rows)
    cfg = ckpt["config"]
    return dict(
        path=path, name=os.path.basename(path), config=cfg, step=ckpt.get("step"), val_loss=ckpt.get("val_loss"),
        file_mb=os.path.getsize(path) / 1e6, load_s=load_s, params=n_params, emb_params=emb,
        vocab=sp.get_piece_size(), rows=rows, n=len(rows),
        exact=sum(r["outcome"] == "correct" for r in rows),
        minor=sum(r["outcome"] == "minor" for r in rows),
        precision=prec, recall=rec, f05=f05,
        wer_before=before / ref_words, wer_after=after / ref_words,
        error_reduction=(before - after) / before if before else 0.0,
        prefers_correct=sum(r["loss_ref"] < r["loss_src"] for r in rows),
        mean_loss=mean_loss, perplexity=math.exp(mean_loss),
        lat_median=statistics.median(lat), lat_p90=sorted(lat)[int(0.9 * (len(lat) - 1))],
        lat_min=min(lat), lat_max=max(lat), total_s=sum(lat) / 1000,
        tok_s=sum(r["new_tokens"] for r in rows) / (sum(lat) / 1000),
        avg_prompt=statistics.mean(r["prompt_tokens"] for r in rows),
        avg_new=statistics.mean(r["new_tokens"] for r in rows),
        flops_token=2 * (n_params - emb) + 2 * model.token_embedding.weight.numel(),
    )


print(f"\nEvaluating {args.model}")
SEEN = set(json.load(open(args.overlap))["test_ids_in_training_data"]) if args.overlap else set()
main = evaluate(resolve_model_path(args.model))
base = None
if args.baseline.lower() != "none" and os.path.exists(resolve_model_path(args.baseline)):
    print(f"\nEvaluating baseline {args.baseline}")
    base = evaluate(resolve_model_path(args.baseline))

pre = None
if args.pretrain and os.path.exists(args.pretrain):
    ck = torch.load(args.pretrain, map_location="cpu")
    if ck["config"].get("embedding_dim") == main["config"].get("embedding_dim") and \
            ck["config"].get("n_layers") == main["config"].get("n_layers"):
        pre = dict(step=ck["step"] + 1, val_loss=ck["val_loss"], tokens=(ck["step"] + 1) * 32 * 8 * 512)
    del ck


# ----------------------------
# HTML report
# ----------------------------
E = html.escape
OUTCOMES = [("correct", "Exact match"), ("minor", "Correct, minor style"), ("improved", "Improved"),
            ("nogain", "Changed, no gain"), ("unchanged", "Unchanged"), ("worse", "Worse")]


def fmt_int(n):
    return f"{n:,}"


def fmt_m(n):
    return f"{n / 1e6:.1f}M"


def pct(x):
    return f"{100 * x:.0f}%"


def diff_html(pred, ref):
    """Prediction with the words that differ from the answer key highlighted."""
    wp, wr = words(pred), words(ref)
    out = []
    for tag, i1, i2, j1, j2 in difflib.SequenceMatcher(None, wp, wr, autojunk=False).get_opcodes():
        chunk = E(" ".join(wp[i1:i2]))
        if tag == "equal":
            out.append(chunk)
        else:
            if i2 > i1:
                out.append(f'<mark class="bad">{chunk}</mark>')
            if j2 > j1:
                out.append(f'<span class="miss" title="expected">{E(" ".join(wr[j1:j2]))}</span>')
    text = " ".join(out)
    return re.sub(r" ([.,!?;:])", r"\1", text)


def bar_chart_errors(rows):
    """Grouped bars: word errors before (input) and after (Pico output) for each sentence."""
    w, h, pad_l, pad_b, top = 720, 220, 30, 26, 10
    n = len(rows)
    mx = max(max(r["err_before"], r["err_after"]) for r in rows) or 1
    slot = (w - pad_l) / n
    bw = slot * 0.36
    sy = (h - pad_b - top) / mx
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Word errors before and after correction per sentence">']
    for k in range(mx + 1):
        y = h - pad_b - k * sy
        parts.append(f'<line x1="{pad_l}" x2="{w}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="axis" text-anchor="end">{k}</text>')
    for idx, r in enumerate(rows):
        x = pad_l + idx * slot + slot * 0.12
        for j, (val, cls, label) in enumerate([(r["err_before"], "c-before", "input"), (r["err_after"], "c-after", "Pico")]):
            bh = max(val * sy, 1.5 if val == 0 else 0)
            parts.append(f'<rect x="{x + j * (bw + 1):.1f}" y="{h - pad_b - bh:.1f}" width="{bw:.1f}" height="{bh:.1f}" '
                         f'rx="1.5" class="{cls}"><title>#{r["id"]} {label}: {val} word errors</title></rect>')
        if r["id"] % 5 == 0 or r["id"] == 1:
            parts.append(f'<text x="{x + bw:.1f}" y="{h - 8}" class="axis" text-anchor="middle">{r["id"]}</text>')
    parts.append("</svg>")
    return "".join(parts)


def bar_chart_latency(rows, median):
    w, h, pad_l, pad_b, top = 720, 200, 44, 26, 10
    n = len(rows)
    mx = max(r["latency_ms"] for r in rows) * 1.1
    step = 100 if mx < 800 else 200 if mx < 1600 else 500
    slot = (w - pad_l) / n
    sy = (h - pad_b - top) / mx
    parts = [f'<svg viewBox="0 0 {w} {h}" class="chart" role="img" aria-label="Latency per sentence">']
    k = 0
    while k <= mx:
        y = h - pad_b - k * sy
        parts.append(f'<line x1="{pad_l}" x2="{w}" y1="{y:.1f}" y2="{y:.1f}" class="grid"/>'
                     f'<text x="{pad_l - 6}" y="{y + 4:.1f}" class="axis" text-anchor="end">{k}</text>')
        k += step
    for idx, r in enumerate(rows):
        x = pad_l + idx * slot + slot * 0.18
        bh = r["latency_ms"] * sy
        parts.append(f'<rect x="{x:.1f}" y="{h - pad_b - bh:.1f}" width="{slot * 0.64:.1f}" height="{bh:.1f}" rx="2" '
                     f'class="c-lat"><title>#{r["id"]}: {r["latency_ms"]:.0f} ms, {r["new_tokens"]} tokens</title></rect>')
        if r["id"] % 5 == 0 or r["id"] == 1:
            parts.append(f'<text x="{x + slot * 0.32:.1f}" y="{h - 8}" class="axis" text-anchor="middle">{r["id"]}</text>')
    y = h - pad_b - median * sy
    parts.append(f'<line x1="{pad_l}" x2="{w}" y1="{y:.1f}" y2="{y:.1f}" class="median"/>')
    parts.append("</svg>")
    return "".join(parts)


def outcome_bar(m):
    counts = [(k, label, sum(r["outcome"] == k for r in m["rows"])) for k, label in OUTCOMES]
    segs = "".join(f'<div class="seg o-{k}" style="flex:{c}" title="{label}: {c}">{c}</div>' for k, label, c in counts if c)
    legend = "".join(f'<span><i class="o-{k}"></i>{label} <b>{c}</b></span>' for k, label, c in counts if c)
    return f'<div class="stack">{segs}</div><div class="legend">{legend}</div>'


def kpi(value, label, sub=""):
    return f'<div class="kpi"><div class="kv">{value}</div><div class="kl">{label}</div><div class="ks">{sub}</div></div>'


m, n = main, main["n"]
cfg = m["config"]
solved = m["exact"] + m["minor"]
kpis = "".join([
    kpi(f"{solved}/{n}", "Sentences fixed", f"{m['exact']} exact" + (f", {m['minor']} with style-only differences" if m['minor'] else " matches with the answer key")),
    kpi(pct(m["error_reduction"]), "Word errors removed", f"WER {pct(m['wer_before'])} → {pct(m['wer_after'])}"),
    kpi(f"{m['f05']:.2f}", "Edit F0.5", f"precision {m['precision']:.2f} · recall {m['recall']:.2f}"),
    kpi(f"{m['lat_median']:.0f} ms", "Median latency", f"CPU · p90 {m['lat_p90']:.0f} ms"),
    kpi(fmt_m(m["params"]), "Parameters", f"{m['file_mb']:.0f} MB on disk"),
    kpi(f"{m['prefers_correct']}/{n}", "Prefers the correct answer", "lower loss on the key than on the input"),
])

arch_rows = [
    ("Architecture", f"GPT-style decoder, {cfg['n_layers']} layers × {cfg['n_heads']} heads"),
    ("Embedding size", cfg["embedding_dim"]),
    ("Context length", f"{cfg['block_size']} tokens"),
    ("Vocabulary", f"{fmt_int(m['vocab'])} SentencePiece BPE tokens"),
    ("Total parameters", fmt_int(m["params"])),
    ("  embeddings (tied with output)", fmt_int(m["emb_params"])),
    ("  transformer blocks", fmt_int(m["params"] - m["emb_params"])),
    ("Compute per token", f"≈ {m['flops_token'] / 1e6:.0f} MFLOPs"),
    ("Weights in memory (fp32)", f"{m['params'] * 4 / 1e6:.0f} MB"),
    ("Checkpoint file", f"{m['name']} · {m['file_mb']:.0f} MB"),
]
train_rows = []
if pre:
    train_rows += [("Pre-training", f"{fmt_int(pre['step'])} steps · ≈ {pre['tokens'] / 1e6:.0f}M tokens"),
                   ("Pre-training val loss", f"{pre['val_loss']:.3f} (perplexity {math.exp(pre['val_loss']):.1f})")]
if m["step"] is not None:
    train_rows.append(("Grammar fine-tuning", f"{fmt_int(m['step'] + 1)} steps"))
if m["val_loss"] is not None:
    train_rows.append(("Fine-tuning val loss", f"{m['val_loss']:.3f} (perplexity {math.exp(m['val_loss']):.2f})"))
train_rows += [("Test-set loss on answer key", f"{m['mean_loss']:.3f} (perplexity {m['perplexity']:.2f})")]


def table(rows):
    return "".join(f'<tr><th>{E(str(k)).replace("  ", "&nbsp;&nbsp;&nbsp;")}</th><td>{E(str(v))}</td></tr>' for k, v in rows)


eff_rows = [
    ("Model load time", f"{m['load_s']:.2f} s"),
    ("Median / p90 latency", f"{m['lat_median']:.0f} ms / {m['lat_p90']:.0f} ms"),
    ("Fastest / slowest", f"{m['lat_min']:.0f} ms / {m['lat_max']:.0f} ms"),
    ("Generation speed", f"{m['tok_s']:.0f} tokens/s"),
    ("Avg. prompt / output tokens", f"{m['avg_prompt']:.1f} / {m['avg_new']:.1f}"),
    (f"All {n} sentences", f"{m['total_s']:.1f} s"),
    ("Hardware", f"CPU · {os.cpu_count()} threads · torch {torch.__version__}"),
]

compare = ""
if base:
    def crow(label, a, b, better_high=True, f=str):
        win = (a > b) if better_high else (a < b)
        return (f'<tr><th>{label}</th><td class="{"win" if win else ""}">{f(a)}</td>'
                f'<td class="{"win" if (not win and a != b) else ""}">{f(b)}</td></tr>')
    compare = f"""
<section class="card">
  <h2>Compared with the small model</h2>
  <p class="note">Same test set, same laptop. The small model ({E(base['name'])}) was pre-trained on a CPU with TinyStories only;
  the main model was pre-trained on a GPU with TinyStories + FineWeb-Edu and fine-tuned with synthetic learner mistakes.</p>
  <table class="cmp"><thead><tr><th></th><th>{E(m['name'])}</th><th>{E(base['name'])}</th></tr></thead><tbody>
  {crow("Sentences fixed", solved, base['exact'] + base['minor'], f=lambda v: f"{v}/{n}")}
  {crow("Exact match", m['exact'], base['exact'], f=lambda v: f"{v}/{n}")}
  {crow("Word errors removed", m['error_reduction'], base['error_reduction'], f=pct)}
  {crow("Edit F0.5", m['f05'], base['f05'], f=lambda v: f"{v:.2f}")}
  {crow("Prefers correct answer", m['prefers_correct'], base['prefers_correct'], f=lambda v: f"{v}/{n}")}
  {crow("Test loss", m['mean_loss'], base['mean_loss'], better_high=False, f=lambda v: f"{v:.3f}")}
  {crow("Median latency", m['lat_median'], base['lat_median'], better_high=False, f=lambda v: f"{v:.0f} ms")}
  {crow("Parameters", m['params'], base['params'], better_high=False, f=fmt_m)}
  </tbody></table>
</section>"""

badge = {"correct": "Exact", "minor": "Style only", "improved": "Improved", "nogain": "No gain",
         "unchanged": "Unchanged", "worse": "Worse"}
SEEN_MARK = '<sup title="also in the training data">*</sup>'
sentence_rows = "".join(
    f'<tr><td class="num">{r["id"]}{SEEN_MARK if r["id"] in SEEN else ""}</td>'
    f'<td>{E(r["src"])}</td><td>{E(r["ref"])}</td>'
    f'<td>{diff_html(r["pred"], r["ref"])}</td><td><span class="badge o-{r["outcome"]}">{badge[r["outcome"]]}</span></td>'
    f'<td class="num">{r["err_before"]}→{r["err_after"]}</td><td class="num">{r["loss_ref"]:.2f}</td>'
    f'<td class="num">{r["latency_ms"]:.0f}</td></tr>' for r in m["rows"])

weak = [r for r in m["rows"] if r["outcome"] in ("unchanged", "worse", "nogain")]
summary = (f"Pico fixed <b>{solved} of {n}</b> sentences completely and removed <b>{pct(m['error_reduction'])}</b> of all "
           f"word errors, running in a median <b>{m['lat_median']:.0f} ms</b> per sentence on a laptop CPU with only "
           f"<b>{fmt_m(m['params'])}</b> parameters. On <b>{m['prefers_correct']} of {n}</b> sentences it assigns the "
           f"answer key a higher probability than the original mistake.")
if SEEN:
    unseen = [r for r in m["rows"] if r["id"] not in SEEN]
    fixed_unseen = sum(r["outcome"] in ("correct", "minor") for r in unseen)
    summary += (f" <b>Note:</b> {len(SEEN)} of the {n} test sentences (or close variants, marked * below) also appear "
                f"in the training data; on the <b>{len(unseen)} unseen</b> sentences Pico fixed <b>{fixed_unseen}</b> "
                f"({pct(fixed_unseen / len(unseen)) if unseen else '–'}).")

LOGO = ('<svg viewBox="0 0 128 128" width="44" height="44" aria-hidden="true">'
        '<rect width="128" height="128" rx="30" fill="#111"/><rect x="30" y="38" width="15" height="62" rx="7.5" fill="#fff"/>'
        '<circle cx="60" cy="58" r="18" fill="none" stroke="#fff" stroke-width="15"/>'
        '<rect x="88" y="55" width="13" height="28" rx="3" fill="#fff" fill-opacity=".55"/></svg>')

page = f"""<!doctype html>
<html lang="en"><head><meta charset="utf-8"><meta name="viewport" content="width=device-width, initial-scale=1">
<title>Pico Model Report</title>
<style>
:root {{ --bg:#f6f7fb; --card:#ffffff; --text:#1d2130; --muted:#646b80; --line:#e3e6ee; --brand:#5b57e8; --accent:#f2b01e;
  --good:#1f9d63; --good-soft:#d8f3e6; --mid:#4f8fe8; --mid-soft:#dde9fb; --warn:#c98a0c; --warn-soft:#fbefd2;
  --bad:#d6455d; --bad-soft:#fadde2; --neutral:#9aa1b2; --neutral-soft:#eceef3; --before:#c3c8d4; }}
@media (prefers-color-scheme: dark) {{ :root:not([data-theme="light"]) {{ --bg:#0f1117; --card:#171a23; --text:#e7e9f0;
  --muted:#9aa1b5; --line:#2a2f3c; --brand:#8b88ff; --good:#3ccf8e; --good-soft:#163828; --mid:#6aa6ff; --mid-soft:#17284a;
  --warn:#e6ad3a; --warn-soft:#3a2c10; --bad:#f06a80; --bad-soft:#41202a; --neutral:#7d8497; --neutral-soft:#252a36; --before:#4a5163; }} }}
:root[data-theme="dark"] {{ --bg:#0f1117; --card:#171a23; --text:#e7e9f0; --muted:#9aa1b5; --line:#2a2f3c; --brand:#8b88ff;
  --good:#3ccf8e; --good-soft:#163828; --mid:#6aa6ff; --mid-soft:#17284a; --warn:#e6ad3a; --warn-soft:#3a2c10; --bad:#f06a80;
  --bad-soft:#41202a; --neutral:#7d8497; --neutral-soft:#252a36; --before:#4a5163; }}
* {{ box-sizing:border-box; }}
body {{ margin:0; background:var(--bg); color:var(--text); font:15px/1.55 "Segoe UI", system-ui, -apple-system, Roboto, sans-serif; }}
main {{ max-width:1100px; margin:0 auto; padding:28px 16px 48px; }}
header {{ display:flex; align-items:center; gap:14px; margin-bottom:6px; }}
h1 {{ font-size:26px; margin:0; letter-spacing:-.3px; }}
h2 {{ font-size:17px; margin:0 0 12px; }}
.meta {{ color:var(--muted); font-size:13px; }}
.summary {{ font-size:16px; margin:18px 0 22px; max-width:900px; }}
.kpis {{ display:grid; grid-template-columns:repeat(auto-fit, minmax(160px, 1fr)); gap:12px; margin-bottom:18px; }}
.kpi, .card {{ background:var(--card); border:1px solid var(--line); border-radius:14px; }}
.kpi {{ padding:14px 16px; }}
.kv {{ font-size:28px; font-weight:700; color:var(--brand); font-variant-numeric:tabular-nums; }}
.kl {{ font-weight:600; font-size:13px; }}
.ks {{ color:var(--muted); font-size:12px; }}
.card {{ padding:18px 20px; margin-bottom:18px; overflow:hidden; }}
.grid2 {{ display:grid; grid-template-columns:1fr 1fr; gap:18px; }}
.grid2 .card {{ margin-bottom:0; }}
@media (max-width:760px) {{ .grid2 {{ grid-template-columns:1fr; }} }}
table {{ width:100%; border-collapse:collapse; font-size:14px; }}
th, td {{ text-align:left; padding:6px 8px; border-bottom:1px solid var(--line); vertical-align:top; }}
.kv-table th {{ color:var(--muted); font-weight:500; width:48%; }}
.kv-table td {{ font-variant-numeric:tabular-nums; }}
.cmp td {{ font-variant-numeric:tabular-nums; }}
.cmp td.win {{ color:var(--good); font-weight:700; }}
.note {{ color:var(--muted); font-size:13px; margin:-4px 0 12px; }}
.chart {{ width:100%; height:auto; display:block; }}
.grid {{ stroke:var(--line); stroke-width:1; }}
.axis {{ fill:var(--muted); font-size:11px; }}
.strong {{ fill:var(--text); font-weight:600; }}
.c-before {{ fill:var(--before); }} .c-after {{ fill:var(--brand); }} .c-lat {{ fill:var(--mid); }}
.median {{ stroke:var(--accent); stroke-width:2; stroke-dasharray:5 4; }}
.legend {{ display:flex; flex-wrap:wrap; gap:6px 16px; font-size:13px; color:var(--muted); margin-top:10px; }}
.legend i {{ display:inline-block; width:10px; height:10px; border-radius:3px; margin-right:6px; vertical-align:-1px; }}
.stack {{ display:flex; height:34px; border-radius:9px; overflow:hidden; gap:2px; }}
.seg {{ display:flex; align-items:center; justify-content:center; color:#fff; font-weight:700; font-size:13px; min-width:22px; }}
.o-correct {{ background:var(--good); }} .o-minor {{ background:#5fbf8f; }} .o-improved {{ background:var(--mid); }}
.o-nogain {{ background:var(--warn); }} .o-unchanged {{ background:var(--neutral); }} .o-worse {{ background:var(--bad); }}
.badge {{ display:inline-block; padding:1px 8px; border-radius:20px; font-size:12px; font-weight:600; color:#fff; white-space:nowrap; }}
.sentences {{ overflow-x:auto; }}
.sentences table {{ min-width:860px; }}
.sentences th {{ font-size:12px; color:var(--muted); font-weight:600; text-transform:uppercase; letter-spacing:.4px; }}
.num {{ text-align:right; font-variant-numeric:tabular-nums; white-space:nowrap; }}
mark.bad {{ background:var(--bad-soft); color:var(--bad); border-radius:4px; padding:0 2px; }}
.miss {{ color:var(--good); text-decoration:underline dotted; font-size:12px; margin-left:2px; }}
.miss::before {{ content:"→ "; }}
ul.limits {{ margin:0; padding-left:20px; }} ul.limits li {{ margin-bottom:6px; }}
footer {{ color:var(--muted); font-size:12px; margin-top:8px; }}
</style></head>
<body><main>
<header>{LOGO}<div><h1>Pico Model Report</h1>
<div class="meta">Grammar correction · {E(m['name'])} · {n} test sentences · {datetime.now():%d %b %Y, %H:%M}</div></div></header>
<p class="summary">{summary}</p>
<div class="kpis">{kpis}</div>

<section class="card">
  <h2>Correction outcomes</h2>
  {outcome_bar(m)}
</section>

<section class="card">
  <h2>Word errors per sentence: input vs. Pico output</h2>
  <p class="note">Word-level edit distance to the answer key. Grey = the original sentence, purple = after Pico. Lower is better; 0 means identical to the key.</p>
  {bar_chart_errors(m['rows'])}
  <div class="legend"><span><i style="background:var(--before)"></i>Input (before)</span><span><i style="background:var(--brand)"></i>Pico output (after)</span></div>
</section>

<div class="grid2">
  <section class="card"><h2>Model</h2><table class="kv-table">{table(arch_rows)}</table></section>
  <section class="card"><h2>Training &amp; loss</h2><table class="kv-table">{table(train_rows)}</table>
    <h2 style="margin-top:18px">Correction quality</h2><table class="kv-table">{table([
        ("Exact match", f"{m['exact']}/{n} ({pct(m['exact'] / n)})"),
        ("Exact or style-only difference", f"{solved}/{n} ({pct(solved / n)})"),
        ("Word error rate before → after", f"{pct(m['wer_before'])} → {pct(m['wer_after'])}"),
        ("Edit precision / recall / F0.5", f"{m['precision']:.2f} / {m['recall']:.2f} / {m['f05']:.2f}"),
    ])}</table></section>
</div>
<div style="height:18px"></div>

<section class="card">
  <h2>Latency per sentence</h2>
  <p class="note">Wall-clock time to generate the full correction (greedy decoding, CPU, median of {args.runs} runs).</p>
  {bar_chart_latency(m['rows'], m['lat_median'])}
  <div class="legend"><span><i style="background:var(--mid)"></i>Latency per sentence (ms)</span>
  <span><i style="background:var(--accent)"></i>Median {m['lat_median']:.0f} ms</span></div>
  <table class="kv-table" style="margin-top:12px">{table(eff_rows)}</table>
</section>
{compare}
<section class="card sentences">
  <h2>All {n} sentences</h2>
  <p class="note">Highlighted words differ from the answer key; <span class="miss">expected</span> shows what the key has instead.
  Errors = word edit distance to the key, before → after. Loss = cross-entropy of the answer key (lower = the model finds it more natural).</p>
  <table><thead><tr><th class="num">#</th><th>Input</th><th>Answer key</th><th>Pico output</th><th>Result</th>
  <th class="num">Errors</th><th class="num">Loss</th><th class="num">ms</th></tr></thead><tbody>{sentence_rows}</tbody></table>
</section>

<section class="card">
  <h2>How to read this</h2>
  <ul class="limits">
    <li><b>Exact match</b> is strict: a valid correction that is worded differently from the key (e.g. a different but correct tense) counts as a miss, so the real usefulness is higher than this number.</li>
    <li><b>Edit F0.5</b> compares the word changes Pico made with the changes in the key, weighting precision twice as much as recall (the usual grammar-correction metric: a wrong change is worse than a missed one).</li>
    <li>The test set is small ({n} sentences), so each sentence is worth {100 / n:.1f} percentage points.</li>
    <li>{len(weak)} sentence(s) were left unchanged or not improved. These are mostly harder rules (e.g. {E(", ".join(r['src'] for r in weak[:3])) or "none"}): adding examples of these patterns to the synthetic training data is the next step.</li>
  </ul>
</section>
<footer>Generated by <code>python -m evaluation.evaluate</code> · inputs normalized (curly quotes → straight) before scoring.</footer>
</main></body></html>"""

os.makedirs(os.path.dirname(args.out) or ".", exist_ok=True)
with open(args.out, "w", encoding="utf-8") as f:
    f.write(page)
with open(os.path.splitext(args.out)[0] + ".json", "w", encoding="utf-8") as f:
    json.dump({"model": main, "baseline": base, "pretrain": pre}, f, indent=1, default=str)
print(f"\n{solved}/{n} fixed ({main['exact']} exact) | word errors removed {pct(main['error_reduction'])} | "
      f"F0.5 {main['f05']:.2f} | median {main['lat_median']:.0f} ms")
print(f"Report: {args.out}")
