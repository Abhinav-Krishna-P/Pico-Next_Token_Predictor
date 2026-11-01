# Pico – Next Token Predictor

Pico is a tiny GPT-style language model built from scratch in PyTorch, made for learning how LLMs work.

Give it a word, and it predicts the next word. Repeat that, and it writes text.

![Pico overview](assets/pico_overview.png)

## Model

| | |
|---|---|
| Parameters | 28,917 |
| Vocabulary | 53 words |
| Context length | 6 words |
| Layers / heads | 2 / 2 |
| Embedding size | 32 |

## Files

- `main.py` – data, tokenizer, model, training and text generation (fully commented)
- `transformer_block.py` – self-attention, multi-head attention, feed-forward and transformer block
- `pico.pt` – the trained model weights

## Run

```bash
pip install torch
python main.py            # trains once and saves pico.pt, then loads it on later runs
python main.py --retrain  # train again from scratch
```

Example output:

```
abhinav krishna p <END> i am working as a software engineer <END> working for the last
```
