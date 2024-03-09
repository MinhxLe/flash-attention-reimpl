import torch
from model import Attention


def test_attention_shape():
    batch, seq_len, embed_dim = 8, 32, 4
    x = torch.rand(batch, seq_len, embed_dim)
    attention = Attention(embed_dim, 2, seq_len)
    out = attention(x)
    assert out.shape == (8, 32, 4)
