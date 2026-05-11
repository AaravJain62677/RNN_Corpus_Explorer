import torch
from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(self, data, sequence_length):

        self.data = data

        self.sequence_length = sequence_length

    def __len__(self):

        return len(self.data) - self.sequence_length

    def __getitem__(self, idx):

        x = self.data[
            idx: idx + self.sequence_length
        ]

        y = self.data[
            idx + 1: idx + self.sequence_length + 1
        ]

        return x, y