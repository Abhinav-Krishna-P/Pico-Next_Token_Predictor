# Pico – Next Token Predictor

Pico is a tiny GPT-style language model built from scratch in PyTorch, made for learning how LLMs work.

Give it some text, and it predicts the next token. Repeat that, and it writes text.

![Pico overview](assets/pico_overview.png)

## Data & tokenizer

- `data/stories.txt` – [TinyStories](https://huggingface.co/datasets/roneneldan/TinyStories) (~27k short stories, 4.4M words of simple grammatical English). Downloaded automatically on the first run; this is what teaches Pico sentences and grammar
- `words_alpha.txt` – ~370k real English words, used only to train the tokenizer so its vocabulary covers rare words
- `corpus.txt` – your own sentences, mixed into the training data
- Tokenizer: SentencePiece BPE with an 8,000-token vocabulary (`tokenizer.model` / `tokenizer.vocab`)
- Each story ends with an end-of-text token; 90/10 train/val split. Encoded tokens are cached in `data/tokens.pt`

## Model

The size is picked automatically based on the hardware:

| | CPU | GPU |
|---|---|---|
| Parameters | ~3.3M | ~13.8M |
| Vocabulary | 8,000 | 8,000 |
| Context length | 64 tokens | 256 tokens |
| Layers / heads | 4 / 6 | 6 / 6 |
| Embedding size | 192 | 384 |
| Batch / steps | 32 / 3000 | 64 / 5000 |

Training uses AdamW with weight decay, linear warmup and cosine LR decay, gradient clipping, dropout, weight tying, and fused multi-head attention (`scaled_dot_product_attention`). The checkpoint with the best validation loss is saved.

## Large pre-training on Google Colab (GPU)

For real fluency, `pico_colab.ipynb` pre-trains a bigger Pico on a GPU:

| | |
|---|---|
| Data | full TinyStories (~2.1M stories) + 300k FineWeb-Edu documents ≈ 965M tokens |
| Tokenizer | 16k SentencePiece BPE with byte fallback (no unknown characters) |
| Model | ~34M parameters, 8 layers / 8 heads, 512 embedding, 512-token context |
| Training | 600M tokens, mixed precision, gradient accumulation, checkpoints every 10 min, auto-resume |

Open the notebook in Colab (**File → Open notebook → GitHub**), pick a **T4 GPU** runtime and run the cells in order. Data and checkpoints are saved to Google Drive (`MyDrive/pico/`), so a disconnect only means re-running the cells. The scripts can also be run directly:

```bash
python prepare_data.py --out-dir data_big                  # download + tokenize (once)
python pretrain.py --data-dir data_big --out pico_big.pt   # pre-train (re-run to resume)
python finetune_grammar.py --base pico_big.pt --out pico_big_grammar.pt --tokenizer-out pico_big_grammar_tokenizer.model --steps 6000
python correct.py --model pico_big_grammar.pt "they was playing in park"
```

## Grammar correction (fine-tuning)

After pre-training, Pico is fine-tuned on `data/train.jsonl` (100k `{"input": wrong, "output": corrected}` pairs). Each pair is trained as `Incorrect: <input> Correct: <output></s>`, with the loss only on the corrected part.

- Symbols the story tokenizer never saw (`( ) & $ %` …) are appended to the vocabulary (`tokenizer_grammar.model`), so all pre-trained token ids stay valid
- Context grows from 64 to 128 tokens (pairs longer than that are skipped, ~76k pairs remain)
- The result is saved to `pico_grammar.pt`; `pico.pt` is left untouched

```bash
python finetune_grammar.py                  # fine-tune for 3000 steps (~70 min on CPU)
python finetune_grammar.py --resume --steps 3000  # continue fine-tuning
python correct.py "they was playing in park"     # correct a sentence
python correct.py                           # interactive mode
```

## Files

- `main.py` – small CPU pre-training: tokenizer, data, training, checkpointing and text generation
- `prepare_data.py` / `pretrain.py` / `pico_colab.ipynb` – large GPU pre-training
- `finetune_grammar.py` – grammar-correction fine-tuning
- `correct.py` – run the grammar-correction model
- `model.py` – the Pico GPT model
- `transformer_block.py` – multi-head causal self-attention, feed-forward and transformer block
- `pico.pt` – the pre-trained checkpoint (weights, optimizer state, config, step, val loss)

## Run

```bash
pip install torch sentencepiece
python main.py                      # trains once and saves pico.pt, then loads it on later runs
python main.py --prompt "The little dog"  # generate from a prompt using the saved model
python main.py --retrain            # train again from scratch
python main.py --resume --steps 2000  # continue training the saved model for 2000 more steps
python main.py --retrain-tokenizer --retrain  # rebuild the tokenizer (and model) from the data files
```
