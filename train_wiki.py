from dataclasses import dataclass
import logging

from torch.utils.data import DataLoader
from data import WikiTextDataset, Split
from model import BlockConfig, NanoGpt, NanoGptConfig
import torch
import torch.optim

device = "cuda"
DEBUG_MODE = False 
logging.basicConfig(level=logging.INFO)

MAX_SEQ_LEN = 100


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 3e-4
    num_epochs: int = 5000 if not DEBUG_MODE else 1


# loading in data
train_dataset = WikiTextDataset(Split.TRAIN)

# training config
train_cfg = TrainConfig()

# model config
model_cfg = NanoGptConfig(
    num_layers=6,
    vocab_size=train_dataset.vocab_size,
    block_config=BlockConfig(
        embed_dim=384,
        num_heads=6,
        seq_len=MAX_SEQ_LEN,
    ),
)

# train data loader
train_dataloader = DataLoader(
    train_dataset,
    batch_size=train_cfg.batch_size,
    shuffle=True,
)


# specify model
model = NanoGpt(model_cfg).to("cuda")
logging.info(f"{sum(p.numel() for p in model.parameters())/1e6} M parameters")

# optimizer
# TODO move training logic to a different module
optimizer = torch.optim.AdamW(model.parameters(), lr=train_cfg.learning_rate)
loss_fn = torch.nn.CrossEntropyLoss()


# train loop


logging.info("Starting training...")
for i in range(train_cfg.num_epochs):
    # TODO add timing
    for batch in train_dataloader:
        model.zero_grad()
        encoded_batch = WikiTextDataset.encode(batch).to("cuda")
        # TODO randomize the slice
        inputs = encoded_batch[:, :MAX_SEQ_LEN]
        outputs = encoded_batch[:, 1 : MAX_SEQ_LEN + 1]
        predict_logits = model(inputs)
        loss = loss_fn(
            predict_logits.reshape(-1, model_cfg.vocab_size),
            outputs.flatten(),
        )
        loss.backward()
        optimizer.step()
        if i % 500 == 0:
            logging.info(f"Epoch: {i}, Loss: {loss.item():.4f}")
    if DEBUG_MODE:
        import ipdb

        ipdb.set_trace()
        break
