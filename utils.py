import os
import torch
import logging

CHECKPOINT_PATH = "./checkpoints"
LOG_PATH = "./logs"


def save_model(model, path, name):
    file_path = os.path.join(CHECKPOINT_PATH, path)
    os.makedirs(file_path, exist_ok=True)
    torch.save(
        model.state_dict(),
        os.path.join(
            file_path,
            f"{name}.pth",
        ),
    )


def get_logger(name, level=logging.INFO):
    # TODO add timestamp
    os.makedirs(LOG_PATH, exist_ok=True)
    logger = logging.getLogger(name)
    logger.setLevel(level)

    file_handler = logging.FileHandler(os.path.join(LOG_PATH, f"{name}.log"), mode="w")
    file_handler.setLevel(level)

    stream_handler = logging.StreamHandler()
    stream_handler.setLevel(level)
    logger.addHandler(stream_handler)
    logger.addHandler(file_handler)

    return logger
