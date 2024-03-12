import pytest
import torch
import torch.nn as nn
from model import Attention, Block, BlockConfig, NanoGpt, NanoGptConfig


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
        BlockConfig(num_heads=num_heads, seq_len=seq_len, embed_dim=embed_dim)
    )
    out = block(x)
    assert out.size() == (4, 32, 8)


def test_nano_gpt_shape():
    config = NanoGptConfig(
        vocab_size=100,
        num_layers=2,
        block_config=BlockConfig(
            embed_dim=16,
            num_heads=2,
            seq_len=128,
        ),
    )
    x = torch.randint(0, 100, size=(32, 128))
    model = NanoGpt(config)
    out = model(x)
    assert out.size() == (32, 128, 100)
