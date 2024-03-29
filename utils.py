import os

CHECKPOINT_PATH = "./checkpoint"


def save_model(model, path, name):
    os.makedirs(os.path.join(CHECKPOINT_PATH, path), exist_ok=True)
    model.save(model.state_dict(), f"{name}.pth")
