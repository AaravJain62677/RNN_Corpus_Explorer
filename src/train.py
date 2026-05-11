import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from config import CONFIG
from preprocess import TextProcessor
from model import RNNModel


def train():
    
    device = torch.device(CONFIG["device"] if torch.cuda.is_available() else "cpu")
    
    processor = TextProcessor(CONFIG["dataset_path"])
    
    sequences = []
    targets = []
    
    for i in range(0, len(processor.text) - CONFIG["sequence_length"]):
        seq = processor.encode(processor.text[i:i + CONFIG["sequence_length"]])
        target = processor.encode(processor.text[i + 1:i + CONFIG["sequence_length"] + 1])
        sequences.append(seq)
        targets.append(target)
    
    sequences = torch.tensor(sequences, dtype=torch.long)
    targets = torch.tensor(targets, dtype=torch.long)
    
    dataset = TensorDataset(sequences, targets)
    
    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )

    model = RNNModel(
        vocab_size=processor.vocab_size,
        embedding_dim=CONFIG["embedding_dim"],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        model_type=CONFIG["model_type"]
    ).to(device)

    criterion = nn.CrossEntropyLoss()

    optimizer = torch.optim.Adam(
        model.parameters(),
        lr=CONFIG["learning_rate"]
    )

    losses = []

    for epoch in range(CONFIG["epochs"]):

        epoch_loss = 0

        for x, y in loader:

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits, _ = model(x)

            loss = criterion(
                logits.reshape(-1, processor.vocab_size),
                y.reshape(-1)
            )

            loss.backward()

            optimizer.step()

            epoch_loss += loss.item()

        avg_loss = epoch_loss / len(loader)

        losses.append(avg_loss)

        print(
            f"Epoch {epoch+1}/{CONFIG['epochs']} | Loss: {avg_loss:.4f}"
        )

    torch.save(
        model.state_dict(),
        "results/checkpoints/model.pth"
    )

    torch.save(
        losses,
        "results/checkpoints/losses.pt"
    )

    print("Training complete.")


if __name__ == "__main__":
    train()