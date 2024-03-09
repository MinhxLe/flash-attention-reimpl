from dataclasses import dataclass
import math
import torch
import torch.nn as nn
from torch.nn import functional as F


class Attention(nn.Module):
    # TODO move parameters to config
    def __init__(self, embed_dim, num_heads, seq_len, dropout=0.0, bias=True):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        # attention parameters
        self.Q = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.K = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.V = nn.Linear(embed_dim, embed_dim, bias=bias)
        self.C_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        # self.dropout = nn.Dropout(p=dropout)

        # lower left triangle since q is row, k is col
        self.register_buffer(
            "attention_mask",
            torch.tril(torch.ones(seq_len, seq_len)).view(seq_len, seq_len),
        )

        self.multihead_attn = nn.MultiheadAttention(
            embed_dim, num_heads, batch_first=True
        )

    def forward(self, x, is_causal=True):
        batch_size, seq_len, embed_dim = x.size()

        expected_y, expected_attention = self.multihead_attn(
            self.Q(x),
            self.K(x),
            self.V(x),
            average_attn_weights=False,
        )
        q = self.Q(x).view(
            batch_size, seq_len, self.num_heads, embed_dim // self.num_heads
        )
        k = self.K(x).view(
            batch_size, seq_len, self.num_heads, embed_dim // self.num_heads
        )
        v = self.V(x).view(
            batch_size, seq_len, self.num_heads, embed_dim // self.num_heads
        )

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
