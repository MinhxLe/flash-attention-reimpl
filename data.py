from enum import StrEnum
import os
import abc
import json
from typing import List
import numpy as np
from pandas.core.computation.parsing import token
import requests
import torch
from tokenizers import CharBPETokenizer, Encoding, Tokenizer
from transformers import GPT2Tokenizer
from datasets import load_dataset
import torch.utils.data
import logging


logging.basicConfig(level=logging.INFO)
DATA_DIR = "./data"


class Split(StrEnum):
    TRAIN = "train"
    TEST = "test"
    VALIDATION = "validation"


class Dataset(abc.ABC, torch.utils.data.Dataset):
    @classmethod
    @abc.abstractmethod
    def encode(cls, text: List[str]) -> torch.Tensor:
        ...

    @classmethod
    @abc.abstractmethod
    def decode(cls, data: torch.Tensor) -> str | List[str]:
        ...


def create_sp_char_dataset(seq_len, batch_size=128):
    # copied from https://github.com/karpathy/nanoGPT
    DATASET_DIR = os.path.join(DATA_DIR, "sp_char")
    input_file_path = os.path.join(DATASET_DIR, "input.txt")
    if not os.path.exists(input_file_path):
        data_url = "https://raw.githubusercontent.com/karpathy/char-rnn/master/data/tinyshakespeare/input.txt"
        with open(input_file_path, "w") as f:
            f.write(requests.get(data_url).text)

    with open(input_file_path, "r") as f:
        data = f.read()
    print(f"length of dataset in characters: {len(data):,}")

    # get all the unique characters that occur in this text
    chars = sorted(list(set(data)))
    vocab_size = len(chars)
    print("all the unique characters:", "".join(chars))
    print(f"vocab size: {vocab_size:,}")

    # create a mapping from characters to integers
    stoi = {ch: i for i, ch in enumerate(chars)}
    itos = {i: ch for i, ch in enumerate(chars)}

    def encode(s) -> list[int]:
        return [stoi[c] for c in s]  # encoder: take a string, output a list of integers

    # create the train and test splits
    encoded_data = encode(data)
    n = len(data)
    n_train = int(0.9 * n)

    train_data = encoded_data[:n_train]
    test_data = encoded_data[n_train:]

    print(f"train has {len(train_data):,} tokens")
    print(f"test has {len(test_data):,} tokens")
    # saving the data

    # export to bin files
    train_ids = np.array(train_data, dtype=np.int16)
    test_ids = np.array(test_data, dtype=np.int16)
    train_ids.tofile(os.path.join(DATASET_DIR, "train.bin"))
    test_ids.tofile(os.path.join(DATASET_DIR, "test.bin"))

    # save the meta information as well, to help us encode/decode later
    meta = {
        "vocab_size": vocab_size,
        "itos": itos,
        "stoi": stoi,
    }
    with open(os.path.join(DATASET_DIR, "meta.json"), "w") as f:
        json.dump(meta, f)

    return (
        torch.tensor(train_ids, dtype=torch.long),
        torch.tensor(test_ids, dtype=torch.long),
        stoi,
        itos,
    )


class WikiTextDataset(Dataset):
    DATASET_DIR = f"{DATA_DIR}/wikitext"

    @classmethod
    @property
    def tokenizer(cls):
        tokenizer = GPT2Tokenizer.from_pretrained("openai-community/gpt2")
        tokenizer.pad_token = tokenizer.eos_token
        return tokenizer

    @classmethod
    @property
    def vocab_size(cls) -> int:
        return cls.tokenizer.vocab_size

    def __init__(self, split: Split):
        dataset = load_dataset(
            "wikitext",
            "wikitext-103-v1",
            split=str(split),
        )["text"]
        # TODO remove all empty and headers
        dataset = [d for d in dataset if len(d) > 0]
        self._dataset = dataset
        logging.info(f"Initialized WikiText {split} dataset")
        logging.info(f"dataset length: {len(dataset)}")
        logging.info(f"max sequence length: {max((len(d) for d in dataset))}")

    @classmethod
    def encode(cls, text: List[str]) -> torch.Tensor:
        embeddings = cls.tokenizer(text, padding="longest")["input_ids"]
        return torch.stack([torch.tensor(e, dtype=torch.long) for e in embeddings])

    @classmethod
    def decode(cls, data: torch.Tensor) -> List[str]:
        return cls.tokenizer.batch_decode(data.tolist())

    def __len__(self):
        return len(self._dataset)

    def __getitem__(self, index):
        return self._dataset[index]
