import random
from dataclasses import dataclass
import logging
from data import create_sp_char_dataset
from model import BlockConfig, NanoGpt, NanoGptConfig
import torch
import torch.optim
import torch.nn.functional as F

device = "cuda"
DEBUG_MODE = False
logging.basicConfig(level=logging.INFO)


@dataclass
class TrainConfig:
    batch_size: int = 32
    learning_rate: float = 3e-4
    num_epochs: int = 5000 if not DEBUG_MODE else 1


# training config
train_cfg = TrainConfig()

# model config
model_cfg = NanoGptConfig(
    num_layers=6,
    vocab_size=65,
    block_config=BlockConfig(
        embed_dim=384,
        num_heads=6,
        seq_len=256,
    ),
)


# loading in data
train_data, test_data, stoi, itos = create_sp_char_dataset(
    train_cfg.batch_size * (model_cfg.block_config.seq_len + 1)
)


def decode(data):
    return "".join([itos[x] for x in data])


# specify model
model = NanoGpt(model_cfg).to("cuda")
logging.info(f"{sum(p.numel() for p in model.parameters())/1e6} M parameters")

# optimizer
# TODO move training logic to a different module
optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()


# train loop


def sample_data(data):
    seq_len = model_cfg.block_config.seq_len
    (total_length,) = data.size()
    # random set of index + range, not sure if this is actually more performant
    random_slices = torch.randint(0, total_length - seq_len, (train_cfg.batch_size,))[
        :, None
    ] + torch.arange(seq_len + 1)
    return data[random_slices]


@torch.no_grad()
def sample_model(sample_len):
    sample = torch.zeros((1, 1), dtype=torch.long, device="cuda")
    for i in range(sample_len):
        context = sample[:, -model_cfg.block_config.seq_len :]
        logits = model(context)[:, -1, :]
        next_idx = torch.multinomial(F.softmax(logits, dim=-1), num_samples=1)
        sample = torch.cat((sample, next_idx), dim=1)
    return decode(sample[0].tolist())


logging.info("Starting training...")
for i in range(train_cfg.num_epochs):
    # TODO add timing
    batch_data = sample_data(train_data)
    input = batch_data[:, :-1].to("cuda")
    output = batch_data[:, 1:].to("cuda")
    model.zero_grad()
    # we do not have source of truth for the last entry
    predicted_output_logits = model(input)
    loss = loss_fn(
        predicted_output_logits.reshape(-1, model_cfg.vocab_size),
        output.flatten(),
    )
    loss.backward()
    optimizer.step()
    if i % 500 == 0:
        logging.info(f"Epoch: {i}, Loss: {loss.item():.4f}")
        logging.info(sample_model(256))
    if DEBUG_MODE:
        import ipdb

        ipdb.set_trace()
        break
