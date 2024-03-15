from dataclasses import dataclass
import math
import torch
from torch.autograd import forward_ad
import torch.nn as nn
from torch.nn import GELU, Sequential, functional as F


@dataclass
class BlockConfig:
    embed_dim: int
    num_heads: int
    seq_len: int
    attn_dropout: float = 0.0
    resid_dropout: float = 0.0
    bias: bool = True


@dataclass
class NanoGptConfig:
    vocab_size: int
    num_layers: int
    block_config: BlockConfig


class Attention(nn.Module):
    # TODO move parameters to cfg
    def __init__(
        self,
        embed_dim,
        num_heads,
        seq_len,
        attn_dropout=0.0,
        resid_dropout=0.0,
        bias=True,
    ):
        super().__init__()
        assert embed_dim % num_heads == 0

        self.num_heads = num_heads
        # attention parameters
        self.C_attn = nn.Linear(embed_dim, embed_dim * 3, bias=bias)
        self.C_proj = nn.Linear(embed_dim, embed_dim, bias=bias)
        # regularization
        self.attn_dropout = nn.Dropout(p=attn_dropout)
        self.resid_dropout = nn.Dropout(p=attn_dropout)
        self.register_buffer("tril", torch.tril(torch.ones(seq_len, seq_len)))

    def forward(self, x, is_causal=True):
        batch_size, seq_len, embed_dim = x.size()
        head_size = embed_dim // self.num_heads

        q, k, v = self.C_attn(x).split(embed_dim, dim=2)  # (B,S,D*3)->(B,S,D)x3
        q = q.view(batch_size, seq_len, self.num_heads, head_size).transpose(1, 2)
        k = k.view(batch_size, seq_len, self.num_heads, head_size).transpose(1, 2)
        v = v.view(batch_size, seq_len, self.num_heads, head_size).transpose(1, 2)

        attn = q @ k.transpose(2, 3) / math.sqrt(head_size)
        if is_causal:
            attn = attn.masked_fill(self.tril[:seq_len, :seq_len] == 0, float("-inf"))
        attn = torch.softmax(attn, dim=-1)
        attn = self.attn_dropout(attn)

        y = attn @ v
        y = y.transpose(1, 2).reshape(batch_size, seq_len, embed_dim)
        return self.resid_dropout(y)

        # TODO this implementation was wrong. figure out why
        # q = q.view(batch_size,k seq_len, self.num_heads, embed_dim // self.num_heads)
        # k = k.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads)
        # v = v.view(batch_size, seq_len, self.num_heads, embed_dim // self.num_heads)
        #
        # # batch_size, head, seq_len, seq_len
        # # where ij is q_i*k_j
        # attention: torch.Tensor = torch.einsum("bihd,bjhd->bhij", q, k) * (
        #     1.0 / math.sqrt(embed_dim // self.num_heads)
        # )
        # if is_causal:
        #     attention = attention.masked_fill(
        #         self.tril[:seq_len, :seq_len] == 0, float("-inf")
        #     )
        #
        # # softmax over the last dimension which is over qk_*
        # attention = torch.softmax(attention, dim=-1)
        # attention = self.attn_dropout(attention)
        # # TODO dropout?
        #
        # y = (
        #     torch.einsum("bhij,bihd->bihd", attention, v)
        #     .contiguous()
        #     .view(batch_size, seq_len, embed_dim)
        # )
        # return self.resid_dropout(self.C_proj(y))


class MLP(nn.Module):
    def __init__(self, embed_dim, dropout=0.0):
        super().__init__()
        self.mlp = nn.Sequential(
            nn.Linear(embed_dim, embed_dim * 4),
            nn.ReLU(),
            nn.Linear(embed_dim * 4, embed_dim),
            nn.Dropout(dropout),
        )

    def forward(self, x):
        return self.mlp(x)


class Block(nn.Module):
    def __init__(self, cfg: BlockConfig):
        super().__init__()
        # normalized on last dimiension
        self.ln1 = nn.LayerNorm(normalized_shape=cfg.embed_dim)
        self.ln2 = nn.LayerNorm(normalized_shape=cfg.embed_dim)
        self.attn = Attention(
            embed_dim=cfg.embed_dim,
            num_heads=cfg.num_heads,
            seq_len=cfg.seq_len,
        )
        self.mlp = MLP(embed_dim=cfg.embed_dim, dropout=cfg.resid_dropout)

    def forward(self, x):
        x = self.attn(self.ln1(x)) + x
        return x + self.mlp(self.ln2(x))


class NanoGpt(nn.Module):
    def __init__(self, cfg: NanoGptConfig):
        super().__init__()
        self.config = cfg
        self.pos_embed = nn.Embedding(
            cfg.block_config.seq_len,
            cfg.block_config.embed_dim,
        )
        self.token_embed = nn.Embedding(
            cfg.vocab_size,
            cfg.block_config.embed_dim,
        )
        self.blocks = nn.Sequential(
            *[Block(cfg.block_config) for _ in range(cfg.num_layers)]
        )
        self.ln = nn.LayerNorm(normalized_shape=cfg.block_config.embed_dim)
        # TODO why is bias false here?
        self.out = nn.Linear(cfg.block_config.embed_dim, cfg.vocab_size, bias=False)

    def forward(self, x):
        batch, seq_len = x.size()
        assert seq_len <= self.config.block_config.seq_len
        # absolute positioning, adding extra dim for
        position = torch.arange(0, seq_len, dtype=torch.long, device="cuda")[None, :]
        x = self.pos_embed(position) + self.token_embed(x)
        x = self.blocks(x)
        return self.out(self.ln(x))
