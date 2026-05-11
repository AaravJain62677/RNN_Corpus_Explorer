import torch
import torch.nn as nn   
class RNNModel(nn.Module):

    def __init__(
        self,
        vocab_size,
        embedding_dim,
        hidden_size,
        num_layers,
        model_type="lstm"
    ):

        super().__init__()

        self.embedding = nn.Embedding(
            vocab_size,
            embedding_dim
        )

        if model_type == "rnn":

            self.rnn = nn.RNN(
                embedding_dim,
                hidden_size,
                num_layers,
                batch_first=True
            )

        elif model_type == "gru":

            self.rnn = nn.GRU(
                embedding_dim,
                hidden_size,
                num_layers,
                batch_first=True
            )

        else:

            self.rnn = nn.LSTM(
                embedding_dim,
                hidden_size,
                num_layers,
                batch_first=True
            )

        self.fc = nn.Linear(
            hidden_size,
            vocab_size
        )

    def forward(self, x, hidden=None):

        x = self.embedding(x)

        output, hidden = self.rnn(x, hidden)

        logits = self.fc(output)

        return logits, hidden