from torch.utils.data import Dataset


class TextDataset(Dataset):

    def __init__(
        self,
        data,
        sequence_length,
        stride=128
    ):

        self.data = data

        self.sequence_length = sequence_length

        self.stride = stride

    def __len__(self):

        return (
            len(self.data) - self.sequence_length
        ) // self.stride

    def __getitem__(self, idx):

        start = idx * self.stride

        x = self.data[
            start:start + self.sequence_length
        ]

        y = self.data[
            start + 1:start + self.sequence_length + 1
        ]

        return x, y