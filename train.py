import random
from dataclasses import dataclass
import logging
from data import create_sp_char_dataset
from model import BlockConfig, NanoGpt, NanoGptConfig
import torch
import torch.optim

TEST_MODE = False
logging.basicConfig(level=logging.INFO)


@dataclass
class TrainConfig:
    batch_size: int = 16
    learning_rate: float = 0.001
    num_epochs: int = 5000 if not TEST_MODE else 1


# training config
train_cfg = TrainConfig()

# model config
model_cfg = NanoGptConfig(
    num_layers=4,
    vocab_size=65,
    block_config=BlockConfig(embed_dim=64, num_heads=4, seq_len=32),
)


# loading in data
train_data, test_data, stoi, itos = create_sp_char_dataset(
    train_cfg.batch_size * (model_cfg.block_config.seq_len + 1)
)

# specify model
model = NanoGpt(model_cfg)

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
    ] + torch.arange(seq_len)
    return data[random_slices]


def sample_model(sample_len):
    sample = [random.randint(0, 66)]
    for i in range(sample_len):
        context = torch.tensor(sample, dtype=torch.int)[
            None, i : i + model_cfg.block_config.seq_len
        ]
        sample.append(model(context)[0, -1, :].argmax().item())
    return "".join([itos[x] for x in sample])


logging.info("Starting training...")
for i in range(train_cfg.num_epochs):
    batch_data = sample_data(train_data)
    input = batch_data[:, :-1]
    output = batch_data[:, -1]
    model.zero_grad()
    predicted_output_logits = model(input)[:, -1]
    loss = loss_fn(predicted_output_logits, output)
    loss.backward()
    optimizer.step()
    if i%100 == 0:
        logging.info(f"Epoch: {i}, Loss: {loss.item():.4f}")
        logging.info(sample_model(32))
