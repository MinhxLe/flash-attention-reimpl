from data import create_sp_char_dataset


def test_create_sp_char_dataset():
    train, test, stoi, itos = create_sp_char_dataset(128, 16)
