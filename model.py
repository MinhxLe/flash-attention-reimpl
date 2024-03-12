from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import GELU, Sequential, functional as F


@dataclass
class Config:
    num_layers: int
    embed_dim: int
    num_heads: int
    seq_len: int
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    bias: bool = True


class Attention(nn.Module):
    # TODO move parameters to config
    def __init__(self, embed_dim, num_heads, seq_len, dropout=0.0, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        # attention parameters
        self.C_attn = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.C_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        # self.dropout = nn.Dropout(p=dropout)

        # lower left triangle since q is row, k is col
        self.register_buffer(
            "attention_mask",
            torch.tril(torch.ones(seq_len, seq_len)).view(seq_len, seq_len),
        )

    def forward(self, x, is_causal=True):
        batch_size, seq_len, embed_dim = x.size()

        q, k, v = self.C_attn(x).split(embed_dim, dim=2)  # (B,S,D*3)->(B,S,D)x3
        q = q.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads)
        k = k.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads)
        v = v.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads)

        # batch_size, head, seq_len, seq_len
        # where ij is q_i*k_j
        attention: torch.Tensor = torch.einsum("bihd,bjhd->bhij", q, k) * (
            1.0 / math.sqrt(embed_dim // self.num_heads)
        )

        if is_causal:
            attention = attention.masked_fill(
                self.attention_mask[None, None, :seq_len, :seq_len] == 0, -math.inf
            )

        # softmax over the last dimension which is over qk_*
        attention = torch.softmax(attention, dim=-1)
        # TODO dropout?

        y = (
            torch.einsum("bhij,bihd->bihd", attention, v)
            .contiguous()
            .view(batch_size, seq_len, embed_dim)
        )
        return self.C_proj(y)


class MLP(nn.Module):
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.GELU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    def __init__(self, config: Config):
        super().__init__()
        # normalized on last dimiension
        self.ln1 = nn.LayerNorm(normalized_shape=config.embed_dim)
        self.ln2 = nn.LayerNorm(normalized_shape=config.embed_dim)
        self.attn = Attention(
            embed_dim=config.embed_dim,
            num_heads=config.num_heads,
            seq_len=config.seq_len,
        )
        self.mlp = MLP(embed_dim=config.embed_dim, dropout=config.resid_dropout)

    def forward(self, x):
        x = self.attn(self.ln1(x)) + x
        return x + self.mlp(self.ln2(x))
