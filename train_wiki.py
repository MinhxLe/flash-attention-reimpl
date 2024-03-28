from dataclasses import dataclass
import logging

from torch.utils.data import DataLoader
from data import WikiTextDataset, Split
from model import BlockConfig, NanoGpt, NanoGptConfig
import torch
import torch.optim

device = "cuda"
torch.random.manual_seed(69)
DEBUG_MODE = False 
logging.basicConfig(level=logging.INFO)
# torch.set_default_dtype(torch.float16)


# configuration from https://openreview.net/pdf?id=COZDy0WYGg


@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 6e-04
    weight_decay: float = 0.01
    num_epochs: int = 5000 if not DEBUG_MODE else 1


# loading in data
train_dataset = WikiTextDataset(Split.TRAIN)

# training config
train_cfg = TrainConfig()

# model config
MAX_SEQ_LEN = 10 if DEBUG_MODE else 1024
model_cfg = NanoGptConfig(
    num_layers=2 if DEBUG_MODE else 12,
    vocab_size=train_dataset.vocab_size,
    block_config=BlockConfig(
        embed_dim=768,
        num_heads=12,
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
# TODO something is off in this math here
logging.info(f"{sum(p.numel() for p in model.parameters())/1e6} M parameters")

# optimizer
# TODO move training logic to a different module
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg.learning_rate,
    weight_decay=train_cfg.weight_decay,
)
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
        logging.info(f"Epoch: {i}, Loss: {loss.item():.4f}")
    if DEBUG_MODE:
        import ipdb

        ipdb.set_trace()
        break
