import datetime
from dataclasses import dataclass
import logging

from torch.utils.data import DataLoader
from data import WikiTextDataset, Split
import utils
from model import BlockConfig, NanoGpt, NanoGptConfig
import torch
import torch.optim

MODEL_NAME = "gpt2_baseline"
torch.random.manual_seed(69)
DEBUG_MODE = True
logging.basicConfig(level=logging.INFO)
# torch.set_default_dtype(torch.float16)


# configuration from https://openreview.net/pdf?id=COZDy0WYGg
@dataclass
class TrainConfig:
    batch_size: int = 64
    learning_rate: float = 6e-04
    weight_decay: float = 0.01
    num_epochs: int = 10 if not DEBUG_MODE else 1


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

PAD_TOKEN = WikiTextDataset.encode(WikiTextDataset.tokenizer.pad_token)[0]
loss_fn = torch.nn.CrossEntropyLoss(reduction="none")


# train loop
logging.info("Starting training...")
for epoch in range(train_cfg.num_epochs):
    # TODO add timing
    count = 0
    epoch_loss = 0
    for batch in train_dataloader:
        encoded_batch = WikiTextDataset.encode(batch).to("cuda")
        for i in range(0, encoded_batch.shape[-1] - MAX_SEQ_LEN):
            count += 1
            model.zero_grad()
            inputs = encoded_batch[:, i : i + MAX_SEQ_LEN]
            outputs = encoded_batch[:, i + 1 : i + MAX_SEQ_LEN + 1]
            predict_logits = model(inputs)
            losses = loss_fn(
# loss function not weighing loss from padding
# mask = torch.ones(model_cfg.vocab_size, device="cuda").masked_fill(
#     torch.arange(model_cfg.vocab_size, device="cuda") == pad_token_idx,
#     0,
# )
                predict_logits.reshape(-1, model_cfg.vocab_size),
                outputs.flatten(),
            )
            padding_mask = (outputs == PAD_TOKEN).flatten()

            losses = losses.masked_fill(padding_mask, 0.0)
            loss = losses.sum() / (~padding_mask).sum()
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()
            if DEBUG_MODE:
                break
        logging.info(
            f"{datetime.datetime.now().isoformat()}, epoch: {epoch}, loss: {epoch_loss/count:.4f}"
        )
        if DEBUG_MODE:
            break
    utils.save_model(model, "gpt2_wt103", f"epoch_{epoch}")
    if DEBUG_MODE:
        import ipdb

        ipdb.set_trace()
