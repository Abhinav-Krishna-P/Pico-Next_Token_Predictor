<p align="center"><picture><source media="(prefers-color-scheme: dark)" srcset="assets/logo_dark.svg"><img src="assets/logo.svg" alt="Pico – next token predictor" width="440"></picture></p>

# Pico – Next Token Predictor

Pico is a tiny GPT-style language model built from scratch in PyTorch, made for learning how LLMs work.
It predicts the next token; repeat that and it writes text. Fine-tuned on grammar data, **Pico v0.3.0
corrects English grammar and answers grammar questions**, running on a laptop CPU:

```
python correct.py "She bought a new furnitures for her room."   ->  She bought a new furniture for her room.
python correct.py "I look forward to meet you."                 ->  I look forward to meeting you.
python correct.py "Despite of the rain, we went out."           ->  Despite the rain, we went out.
python ask.py "What is a noun?"                                 ->  A noun is a naming word. It refers to a person, a place or a thing. …
```

<p align="center"><img src="assets/poster.png" alt="Pico v0.3.0 poster" width="540"></p>

## Pico v0.3.0 at a glance

| | |
|---|---|
| Parameters | **34,120,704** (34.1M) – 8.9M embeddings (tied with the output) + 25.2M transformer |
| Weights | **137 MB** (float32) |
| Architecture | GPT decoder · 8 layers × 8 heads · 512 embedding · 1,024-token context · ≈ 67 MFLOPs per token |
| Tokenizer | SentencePiece BPE, **16,384** tokens, byte fallback |
| Pre-training | **776M tokens** (TinyStories + FineWeb-Edu), 600M trained · 4,577 steps · val loss 2.50 · Kaggle 2× T4, ~4 h |
| Fine-tuning | 100k grammar pairs + 388k synthetic learner mistakes + the [English Practical Grammar dataset](https://github.com/Abhinav-Krishna-P/English-practical-grammer-Dataset) (4,233 conversations + 653 extracted pairs) · 20,000 steps · val loss 0.92 |
| Benchmark (30 sentences) | **18/30** fixed exactly · **54%** of word errors removed · precision **0.83** · F0.5 **0.77** · test loss **0.28** |
| Standard benchmarks (ERRANT F0.5) | BEA-2019 dev **0.16** · CoNLL-2014 **0.19** · JFLEG **0.36** – real learner essays, see [Evaluate](#evaluate) (0.10 / 0.10 / 0.26 without the [edit filter](#edit-filter)) |
| Speed | ~0.4–0.5 s per sentence on a laptop CPU |

### Versions

| | v0.1.0 | v0.2.0 | v0.3.0 |
|---|---|---|---|
| Parameters | 28,917 | 33.9M | **34.1M** |
| Vocabulary | 53 words | 16,384 BPE | 16,384 BPE |
| Layers × heads | 2 × 2 | 8 × 8 | 8 × 8 |
| Training data | 9 sentences | 776M tokens + 485k grammar pairs | + English Practical Grammar dataset |
| Benchmark | – | 11/30 | **18/30** |
| What it does | memorises its corpus | corrects grammar | corrects grammar + **answers grammar questions** |

The v0.1.0 overview is kept in [assets/pico_overview.png](assets/pico_overview.png).

## Project structure

```
Pico/
├── correct.py                  # fix grammar (start here)
├── ask.py                      # ask a grammar question
├── pico/                       # the model
│   ├── model.py                #   Pico GPT: embeddings, transformer blocks, generation, resizing
│   ├── transformer_block.py    #   causal self-attention, feed-forward, transformer block
│   └── grammar.py              #   load a model, correct sentences (+ edit filter), answer questions
├── models/                     # the trained model (not in git, see below)
│   ├── pico_big_grammar_small.pt
│   └── pico_big_grammar_tokenizer.model
├── evaluation/
│   ├── evaluate.py             #   benchmark a model and write a one-page HTML report
│   ├── benchmark.py            #   score on BEA-2019 / CoNLL-2014 / JFLEG with ERRANT
│   ├── m2.py                   #   read M2 correction files, detokenize sentences
│   ├── test_set/               #   30 test sentences (qns.txt) + answer key (ans.txt)
│   └── reports/                #   report.html + report.json, benchmarks/<model>.json
├── training/                   # everything used to train the model
│   ├── prepare_data.py         #   download + tokenize the pre-training data (data/big/)
│   ├── pretrain.py             #   pre-training on a GPU
│   ├── make_synthetic.py       #   synthetic grammar pairs with common learner mistakes
│   ├── extract_grammar_pairs.py#   correction pairs from the English Practical Grammar dataset
│   ├── prepare_learner_data.py #   real learner essays (W&I+LOCNESS, FCE) as training pairs (stage 2)
│   ├── finetune_grammar.py     #   grammar fine-tuning (correction pairs + Question/Answer conversations)
│   ├── export_model.py         #   strip the optimizer state -> ~3x smaller file for inference
│   └── train_small.py          #   small CPU-only model (for learning, v0.1-style)
├── notebooks/
│   └── pico_kaggle.ipynb       # the full training run on a free Kaggle GPU
├── data/
│   ├── train.jsonl             # 100k grammar pairs (not in git)
│   ├── grammar_book.jsonl      # pairs extracted from the grammar dataset (not in git)
│   ├── learner.jsonl           # stage-2 learner essay pairs (not in git)
│   ├── benchmarks/             # BEA-2019 / FCE / CoNLL-2014 / JFLEG downloads (not in git)
│   ├── corpus.txt, words_alpha.txt   # used by train_small.py / the tokenizer
├── assets/                     # logo, poster, v0.1.0 overview
└── requirements.txt
```

All commands are run from the project root. Training scripts are run as modules (`python -m training.<name>`).

## Setup

```bash
python -m venv .venv
.venv\Scripts\activate          # Windows  (Linux/macOS: source .venv/bin/activate)
pip install -r requirements.txt
```

The model files are too big for git: download `pico_big_grammar_small.pt` and `pico_big_grammar_tokenizer.model`
from the Kaggle notebook's **Output** tab into `models/`.

## Use

```bash
python correct.py "they was playing in park"     # correct one sentence
python correct.py                                # interactive mode (empty line to quit)
python correct.py --no-filter "..."              # raw model output, without the edit filter
python ask.py "What is a noun?"                  # grammar question (experimental, see below)
python ask.py                                    # interactive mode
```

`ask.py` is **experimental**: answers are good for questions close to the English Practical Grammar dataset
(*What is a noun?*), but for other questions a 34M-parameter model often repeats itself or makes things up.
Sentence correction (`correct.py`) is the reliable feature.

## Evaluate

```bash
python -m evaluation.evaluate                                    # writes evaluation/reports/report.html
python -m evaluation.evaluate --baseline models/other_model.pt   # compare two models
```

Runs every sentence in `evaluation/test_set/qns.txt` through the model, compares it with `ans.txt` and reports
parameters, tokens, latency, loss / perplexity, exact match, word errors before vs after and edit precision /
recall / F0.5, with a per-sentence breakdown.

| | v0.3.0 + edit filter | v0.3.0 raw | v0.2.0 |
|---|---|---|---|
| Sentences fixed exactly | **18/30** | 18/30 | 11/30 |
| Fixed, on the 19 sentences **not** in the training data | **9/19** | 9/19 | 8/19 |
| Word errors removed | **54%** | 51% | 26% |
| Edit precision / F0.5 | **0.83 / 0.77** | 0.76 / 0.72 | 0.69 / 0.57 |
| Test loss on the answer key | **0.28** | 0.28 | 0.49 |
| Median latency (laptop CPU) | ~0.4–0.5 s | ~0.4–0.5 s | ~0.5 s |

11 of the 30 test sentences (or close variants) also appear in the English Practical Grammar dataset. They are
marked in the report (`data/grammar_book_test_overlap.json`), and the score on the 19 unseen sentences is the
fair comparison.

### Standard benchmarks

30 sentences are too few to measure progress (one sentence = 3 points), so `evaluation/benchmark.py` also scores
the model on the public grammar-correction test sets that research papers report, with
[ERRANT](https://github.com/chrisjbryant/errant) (span-based correction precision / recall / F0.5):

```bash
pip install errant && python -m spacy download en_core_web_sm   # once
python -m evaluation.benchmark                                  # all three sets, ~35 min on a laptop CPU
python -m evaluation.benchmark --sets bea --limit 300           # quick check
python -m evaluation.benchmark --model models/other_model.pt    # another model
```

| Test set | Sentences | Text | Precision | Recall | F0.5 | F0.5 raw |
|---|---|---|---|---|---|---|
| BEA-2019 dev (W&I+LOCNESS) | 4,384 | essays by learners of all levels + native students | 0.25 | 0.06 | **0.16** | 0.10 |
| CoNLL-2014 test | 1,312 | essays by university students (2 references) | 0.34 | 0.07 | **0.19** | 0.10 |
| JFLEG test | 747 | learner sentences, fluency rewrites (4 references) | 0.54 | 0.15 | **0.36** | 0.26 |

*F0.5 raw* is the model output without the [edit filter](#edit-filter) (`--no-filter`).

These numbers are much lower than the 30-sentence test, and they are the honest ones: real learner essays have
long sentences, several errors each and many error types that the synthetic training data does not cover. The
main problem was **precision** – without the filter, 4,363 of the model's 4,883 edits on BEA are wrong, mostly
rewrites of words that were already correct, and it changes 41% of the sentences that need no correction (its
training data had no correct sentences). The strongest error types are capitalization (ORTH, F0.5 0.39 on BEA,
0.75 on JFLEG), determiners, punctuation and verb tense; spelling and verb choice are barely fixed. For scale,
large published systems score roughly 0.6–0.75 F0.5 on these sets.

### Edit filter

`correct.py` (and the benchmark) splits the model's correction into word-level edits and keeps an edit only if

1. it is not a **rewrite**: it changes only punctuation / capitalization, small function words (*a → an*,
   *in → on*, *+ the*) and the form or spelling of the other words (*go → went*, *am knowing → know*,
   *recieve → receive*). Rewrites into an unrelated word (*Psychology → the parts*) were half of the model's
   edits on BEA dev and only 2% of them were right;
2. the model does not clearly reject it: log P(sentence with only this edit) − log P(sentence unchanged)
   must be above −2.

It was tuned on the even sentences of BEA-2019 dev and checked on the odd ones (F0.5 0.10 → 0.17), CoNLL-2014
(0.10 → 0.19) and JFLEG (0.26 → 0.36). The model's own confidence alone helped less (0.15 / 0.16 / 0.34): it is
confident about many of its wrong rewrites. The filter only removes edits, so it cannot fix more errors than the
model finds – that needs better training (see stage 2 under [Training](#training)).

How it works and what to know:

- The test sets are downloaded once into `data/benchmarks/`. They are licensed for **non-commercial research only**,
  so they – and the model's corrections of them – are git-ignored; only the scores
  (`evaluation/reports/benchmarks/<model>.json`, with a per-error-type breakdown) are committed.
- The sentences are word-tokenized (`I do n't know .`); they are detokenized for the model, and in the output
  every word the model left unchanged keeps the original tokens, so an unchanged sentence never gets fake edits.
- Corrections are greedy-decoded like `correct.py` (same output, checked), but with a KV cache and batching:
  ~0.3 s per sentence instead of ~2 s. The raw corrections are cached, so re-scoring only re-runs the edit
  filter (~5 min for all sets; `--no-filter` scores the raw output, `--rerun` regenerates the corrections).
- The BEA-2019 *test* answers are hidden, so the dev set is used. CoNLL-2014 papers usually report the M2 scorer
  and JFLEG papers GLEU; ERRANT numbers are close but not identical to those.

## Training

Everything runs in `notebooks/pico_kaggle.ipynb` on a free Kaggle GPU: import it (**Create → New Notebook → File →
Import Notebook**), enable **GPU T4 x2** and **Internet**, add `train.jsonl` as an input dataset, then **Save Version →
Save & Run All**. It runs in the background for up to 12 hours, stops itself safely before the limit and continues
from its own previous output. Once pre-training is done, adding that output as input (**Add Input → Your Work →
Notebooks**) re-runs only the grammar fine-tuning (~1 hour).

**1. Pre-training** – full TinyStories (2.1M stories, 452M tokens) + 300k FineWeb-Edu documents (328M tokens);
16k SentencePiece BPE tokenizer with byte fallback; 600M training tokens with mixed precision, gradient accumulation,
AdamW, warmup + cosine decay and checkpoints every 10 minutes.

**2. Grammar fine-tuning** – each example is trained with the loss only on the answer part:

- `Incorrect: <wrong sentence> Correct: <corrected sentence></s>` for
  - `data/train.jsonl` – 100k pairs, mostly long Wikipedia-style sentences
  - `make_synthetic.py` – ~388k short sentences broken on purpose with common learner mistakes (a/an, missing
    articles, *dont*, was/were, *he go*, wrong tense, *I am work*, prepositions, plurals, punctuation …)
  - `extract_grammar_pairs.py` – 653 pairs from the dataset's exercises (corrections, prepositions, articles,
    tenses, verb choice, punctuation), repeated x20
- `Question: <question> Answer: <answer></s>` for **all 4,233 conversations** of the English Practical Grammar
  dataset (21 chapters), repeated x10

```bash
python -m training.prepare_data                       # download + tokenize into data/big (once)
python -m training.pretrain                           # pre-train models/pico_big.pt (re-run to resume)
python -m training.make_synthetic --out data/synthetic.jsonl
git clone https://github.com/Abhinav-Krishna-P/English-practical-grammer-Dataset data/english-practical-grammar
python -m training.extract_grammar_pairs              # -> data/grammar_book.jsonl + test-set overlap
python -m training.finetune_grammar --data data/train.jsonl data/synthetic.jsonl "data/grammar_book.jsonl*20" \
    $(for f in data/english-practical-grammar/data/*.jsonl; do echo "$f*10"; done) \
    --block-size 1024 --steps 20000 --batch-size 32  # -> models/pico_big_grammar.pt
python -m training.export_model models/pico_big_grammar.pt models/pico_big_grammar_small.pt
```

**3. Stage 2: real learner essays** (new, not trained yet) – the benchmarks showed that the model rewrites far
too much, because every training input above contains a mistake. Stage 2 continues from the stage-1 model with a
lower learning rate on human-corrected learner writing from the BEA-2019 shared task:

- `prepare_learner_data.py` – W&I+LOCNESS train + FCE train/dev: **61,876 sentence pairs, 36% already correct**
  (input = output), repeated x2. The benchmark test sets (BEA-2019 dev, FCE test, CoNLL-2014, JFLEG) are not used.
- the grammar-dataset correction pairs (x5) and conversations (x2), so stage 1 is not forgotten

```bash
python -m training.prepare_learner_data               # -> data/learner.jsonl (non-commercial licence, git-ignored)
python -m training.finetune_grammar --data "data/learner.jsonl*2" "data/grammar_book.jsonl*5" \
    $(for f in data/english-practical-grammar/data/*.jsonl; do echo "$f*2"; done) \
    --base models/pico_big_grammar_small.pt --out models/pico_big_grammar_learner.pt \
    --block-size 1024 --steps 4000 --batch-size 32 --lr 1e-4
python -m training.export_model models/pico_big_grammar_learner.pt models/pico_big_grammar_small.pt
python -m evaluation.benchmark --rerun                # compare with the scores above
```

This needs a GPU (~45 s per step on a laptop CPU). In the Kaggle notebook it is section **6b**: it runs right
after stage 1, or alone (~30 min) when a previous version with `pico_big_grammar.pt` is added as input.

`python -m training.train_small` trains a small 3.3M-parameter model on TinyStories on a CPU only, for learning
how pre-training works.
