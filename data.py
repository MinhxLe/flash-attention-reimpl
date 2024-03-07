import os
import json
import numpy as np
import requests
import torch
from torch.utils.data import TensorDataset, random_split, DataLoader


DATA_DIR = "./data"


def create_sp_char_dataset(batch_size=128):
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

    def decode(l):
        return "".join(
            [itos[i] for i in l]
        )  # decoder: take a list of integers, output a string

    # create the train and test splits
    encoded_data = encode(data)
    n = len(data)
    n_train = int(0.9 * n)
    train_data, test_data = random_split(
        encoded_data,
        [n - n_train, n_train],
    )

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

    return DataLoader(
        TensorDataset(torch.tensor(train_ids)),
        batch_size=batch_size,
        shuffle=True,
    ), DataLoader(
        TensorDataset(torch.tensor(test_ids)),
        batch_size=batch_size,
        shuffle=True,
    )
