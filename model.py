# model.py – the Pico GPT model, shared by main.py, finetune_grammar.py and correct.py

import os

import torch
import torch.nn as nn
import torch.nn.functional as F

from transformer_block import Block


def find_tokenizer(ckpt, ckpt_path, default):
    """Tokenizer path stored in a checkpoint, falling back to the same file name next to the
    checkpoint (so checkpoints keep working after being moved, e.g. downloaded from Colab)."""
    path = ckpt.get("tokenizer", default)
    if not os.path.exists(path):
        local = os.path.join(os.path.dirname(os.path.abspath(ckpt_path)), os.path.basename(path))
        if os.path.exists(local):
            return local
    return path


class Pico(nn.Module):
    def __init__(self, vocab_size, block_size, embedding_dim, n_heads, n_layers, dropout, **_):
        super().__init__()
        self.block_size = block_size
        self.token_embedding = nn.Embedding(vocab_size, embedding_dim)
        self.position_embedding = nn.Embedding(block_size, embedding_dim)
        self.drop = nn.Dropout(dropout)
        self.blocks = nn.Sequential(*[Block(embedding_dim, block_size, n_heads, dropout) for _ in range(n_layers)])
        self.ln_f = nn.LayerNorm(embedding_dim)
        self.head = nn.Linear(embedding_dim, vocab_size, bias=False)
        self.head.weight = self.token_embedding.weight  # weight tying: saves vocab_size*embedding_dim params
        self.apply(self._init_weights)

    def _init_weights(self, module):
        if isinstance(module, nn.Linear):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)
            if module.bias is not None:
                nn.init.zeros_(module.bias)
        elif isinstance(module, nn.Embedding):
            nn.init.normal_(module.weight, mean=0.0, std=0.02)

    def forward(self, idx, targets=None):
        B, T = idx.shape
        tok_emb = self.token_embedding(idx)
        pos_emb = self.position_embedding(torch.arange(T, device=idx.device))
        x = self.drop(tok_emb + pos_emb)
        x = self.blocks(x)
        x = self.ln_f(x)
        logits = self.head(x)
        loss = None
        if targets is not None:
            B, T, C = logits.shape
            # targets of -100 are ignored (used to train only on the answer part during fine-tuning)
            loss = F.cross_entropy(logits.view(B*T, C), targets.reshape(B*T), ignore_index=-100)
        return logits, loss

    @torch.no_grad()
    def generate(self, idx, max_new_tokens, temperature=0.8, top_k=50, stop_id=None):
        # temperature=0 means greedy decoding (always pick the most likely token)
        for _ in range(max_new_tokens):
            idx_cond = idx[:, -self.block_size:]
            logits, _ = self(idx_cond)
            logits = logits[:, -1, :]
            if temperature == 0:
                next_idx = logits.argmax(dim=-1, keepdim=True)
            else:
                logits = logits / temperature
                if top_k is not None:
                    v, _ = torch.topk(logits, min(top_k, logits.size(-1)))
                    logits[logits < v[:, [-1]]] = -float("inf")
                probs = F.softmax(logits, dim=-1)
                next_idx = torch.multinomial(probs, 1)
            idx = torch.cat((idx, next_idx), dim=1)
            if stop_id is not None and next_idx.item() == stop_id:
                break
        return idx

    def resize(self, vocab_size=None, block_size=None):
        """Grow the vocabulary and/or context length, keeping all trained weights."""
        dim = self.token_embedding.embedding_dim
        if vocab_size is not None and vocab_size > self.token_embedding.num_embeddings:
            old = self.token_embedding.weight.data
            new = nn.Embedding(vocab_size, dim).to(old.device)
            nn.init.normal_(new.weight, mean=0.0, std=0.02)
            new.weight.data[:old.size(0)] = old
            self.token_embedding = new
            self.head = nn.Linear(dim, vocab_size, bias=False).to(old.device)
            self.head.weight = self.token_embedding.weight
        if block_size is not None and block_size > self.block_size:
            old = self.position_embedding.weight.data
            new = nn.Embedding(block_size, dim).to(old.device)
            nn.init.normal_(new.weight, mean=0.0, std=0.02)
            new.weight.data[:old.size(0)] = old
            self.position_embedding = new
            self.block_size = block_size
