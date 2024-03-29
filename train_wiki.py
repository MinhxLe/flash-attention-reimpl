import datetime
from dataclasses import dataclass

from torch.utils.data import DataLoader
from data import WikiTextDataset, Split
import utils
from model import BlockConfig, NanoGpt, NanoGptConfig
import torch
import torch.optim

RUN_NAME = "gpt2_wt103_baseline"
torch.random.manual_seed(69)
DEBUG_MODE = True
logger = utils.get_logger(RUN_NAME)
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
test_dataset = WikiTextDataset(Split.TRAIN)

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
test_dataloader = DataLoader(
    test_dataset,
    batch_size=train_cfg.batch_size,
    shuffle=True,
)


# specify model
model = NanoGpt(model_cfg).to("cuda")
# TODO something is off in this math here
logger.info(f"{sum(p.numel() for p in model.parameters())/1e6} M parameters")

# optimizer
# TODO move training logic to a different module
optimizer = torch.optim.AdamW(
    model.parameters(),
    lr=train_cfg.learning_rate,
    weight_decay=train_cfg.weight_decay,
)

PAD_TOKEN = WikiTextDataset.encode(WikiTextDataset.tokenizer.pad_token)[0]
loss_fn = torch.nn.CrossEntropyLoss(reduction="none")


def get_test_loss(num_batches=None):
    model.eval()
    # TODO abstract out between train and test
    count = 0
    total_loss = 0
    with torch.no_grad():
        for curr_batch, batch in enumerate(test_dataloader):
            encoded_batch = WikiTextDataset.encode(batch).to("cuda")
            for i in range(0, encoded_batch.shape[-1] - MAX_SEQ_LEN):
                inputs = encoded_batch[:, i : i + MAX_SEQ_LEN]
                outputs = encoded_batch[:, i + 1 : i + MAX_SEQ_LEN + 1]
                predict_logits = model(inputs)
                # loss masking out pads
                losses = loss_fn(
                    predict_logits.reshape(-1, model_cfg.vocab_size),
                    outputs.flatten(),
                )
                padding_mask = (outputs == PAD_TOKEN).flatten()
                losses = losses.masked_fill(padding_mask, 0.0)
                loss = losses.sum()
                count += (~padding_mask).sum()
                total_loss += loss.item()
            if num_batches is not None and curr_batch == num_batches:
                break
    model.train()
    return total_loss / count


# train loop
logger.info("Starting training...")
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
            # loss masking out pads
            losses = loss_fn(
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
        logger.info(
            f"{datetime.datetime.now().isoformat()}:TRAIN: epoch: {epoch}, train loss: {epoch_loss/count:.4f}"
        )
        if DEBUG_MODE:
            break
    test_loss = get_test_loss(num_batches=5)
    logger.info(
        f"{datetime.datetime.now().isoformat()}:TEST: epoch: {epoch}, loss: {test_loss:.4f}"
    )
    utils.save_model(model, RUN_NAME, f"epoch_{epoch}")
    if DEBUG_MODE:
        import ipdb

        ipdb.set_trace()
