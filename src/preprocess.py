import torch


class TextProcessor:

    def __init__(self, filepath):

        self.filepath = filepath

        self.text = self.load_text()

        self.chars = sorted(list(set(self.text)))

        self.vocab_size = len(self.chars)

        self.char_to_idx = {
            ch: idx for idx, ch in enumerate(self.chars)
        }

        self.idx_to_char = {
            idx: ch for idx, ch in enumerate(self.chars)
        }

    def load_text(self):

        with open(self.filepath, "r", encoding="utf-8") as f:
            text = f.read()

        return text

    def encode(self, text):

        return [
            self.char_to_idx[ch]
            for ch in text
        ]

    def decode(self, indices):

        return "".join(
            [
                self.idx_to_char[idx]
                for idx in indices
            ]
        )

    def get_tensor(self):

        encoded = self.encode(self.text)

        return torch.tensor(
            encoded,
            dtype=torch.long
        )