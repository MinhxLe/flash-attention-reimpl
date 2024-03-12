import pytest
import torch
import torch.nn as nn
from model import Attention, Block, Config


def test_attention_shape():
    batch, seq_len, embed_dim = 8, 32, 4
    x = torch.rand(batch, seq_len, embed_dim)
    attention = Attention(embed_dim, 2, seq_len)
    out = attention(x)
    assert out.size() == (8, 32, 4)


@pytest.mark.skip()
def test_attention_correctness():
    # TODO this isn't working
    batch, seq_len, num_heads, embed_dim = 4, 32, 1, 8
    x = torch.rand(batch, seq_len, embed_dim)

    attention = Attention(embed_dim, num_heads, seq_len)
    out = attention(x, is_causal=False)

    q = attention.Q(x)
    k = attention.K(x)
    v = attention.V(x)
    torch_attention = nn.MultiheadAttention(embed_dim, num_heads, batch_first=True)
    expected_out, expected_attn_weight = torch_attention(q, k, v)
    assert out.shape == expected_out.shape
    assert torch.allclose(out, expected_out)


def test_block_shape():
    batch, seq_len, num_heads, embed_dim = 4, 32, 1, 8
    x = torch.rand(batch, seq_len, embed_dim)
    block = Block(
        Config(num_heads=num_heads, seq_len=seq_len, embed_dim=embed_dim, num_layers=1)
    )
    out = block(x)
    assert out.size() == (4, 32, 8)
