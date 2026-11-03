# transformer_blocks.py

import torch
import torch.nn as nn
import torch.nn.functional as F

# ----------------------------
# Multi-Head Causal Self-Attention
# ----------------------------
# All heads are computed in one batched matmul (instead of a Python loop over
# separate head modules) and use PyTorch's fused scaled_dot_product_attention,
# which is much faster and uses less memory.
class MultiHeadAttention(nn.Module):
    def __init__(self, embedding_dim, block_size, num_heads, dropout=0.0):
        super().__init__()
        assert embedding_dim % num_heads == 0, "embedding_dim must be divisible by num_heads"
        self.num_heads = num_heads
        self.qkv = nn.Linear(embedding_dim, 3 * embedding_dim, bias=False)  # query, key, value in one projection
        self.proj = nn.Linear(embedding_dim, embedding_dim)
        self.dropout = dropout
        self.resid_dropout = nn.Dropout(dropout)

    def forward(self, x):
        B, T, C = x.shape
        q, k, v = self.qkv(x).split(C, dim=2)
        # (B, T, C) -> (B, n_heads, T, head_size)
        q = q.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        k = k.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        v = v.view(B, T, self.num_heads, C // self.num_heads).transpose(1, 2)
        out = F.scaled_dot_product_attention(
            q, k, v, is_causal=True, dropout_p=self.dropout if self.training else 0.0
        )
        out = out.transpose(1, 2).contiguous().view(B, T, C)  # re-join the heads
        return self.resid_dropout(self.proj(out))

# ----------------------------
# Feed Forward Network
# ----------------------------
class FeedForward(nn.Module):
    def __init__(self, n_embd, dropout=0.0):
        super().__init__()
        self.net = nn.Sequential(
            nn.Linear(n_embd, 4 * n_embd),
            nn.GELU(),
            nn.Linear(4 * n_embd, n_embd),
            nn.Dropout(dropout),
        )
    def forward(self, x):
        return self.net(x)

# ----------------------------
# Transformer Block
# ----------------------------
class Block(nn.Module):
    def __init__(self, embedding_dim, block_size, n_heads, dropout=0.0):
        super().__init__()
        self.sa = MultiHeadAttention(embedding_dim, block_size, n_heads, dropout)
        self.ffwd = FeedForward(embedding_dim, dropout)
        self.ln1 = nn.LayerNorm(embedding_dim)
        self.ln2 = nn.LayerNorm(embedding_dim)

    def forward(self, x):
        x = x + self.sa(self.ln1(x))
        x = x + self.ffwd(self.ln2(x))
        return x
