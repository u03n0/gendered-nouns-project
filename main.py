import torch
import numpy as np
from torch.utils.data import  DataLoader, SubsetRandomSampler
from models import LanguageDataset, GenderedLSTM


if __name__ == "__main__":
    with open("french_clean.txt") as f:
        text = f.readlines()

    # Hyperparameters model
    vocab_size = 70
    window_size = 10
    embedding_dim = 2
    hidden_dim = 16
    dense_dim = 32
    n_layers = 1
    max_norm = 2

    # Training config
    train_val_split = 0.8
    batch_size = 128
    random_state = 13

    torch.manual_seed(random_state)

    dataset = LanguageDataset(text)
    n_epochs = dataset.longest_noun

    n_samples = len(dataset)
    split_idx = int(n_samples * train_val_split)

    train_indices, val_indices = np.arange(split_idx), np.arange(split_idx, n_samples)

    train_dataloader = DataLoader(
                dataset, sampler=SubsetRandomSampler(train_indices), batch_size=batch_size
        )
    val_dataloader = DataLoader(
                dataset, sampler=SubsetRandomSampler(val_indices), batch_size=batch_size
    )

    model = GenderedLSTM(train_dataloader, )
    model.train(train_dataloader, val_dataloader, 20, 32, device='cuda')