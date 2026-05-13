import torch
import torch.nn as nn
from torch.utils.data import DataLoader

from src.config import CONFIG
from src.preprocess import TextProcessor
from src.dataset import TextDataset
from src.model import RNNModel


def train():

    print("Loading dataset...")

    device = torch.device(
        CONFIG["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    processor = TextProcessor(
        CONFIG["dataset_path"]
    )

    data = processor.get_tensor()

    print(f"Dataset Size: {len(data)}")
    print(f"Vocabulary Size: {processor.vocab_size}")

    dataset = TextDataset(
        data=data,
        sequence_length=CONFIG["sequence_length"]
    )

    loader = DataLoader(
        dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )

    print("Initializing model...")

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

    print("Starting training...\n")

    for epoch in range(CONFIG["epochs"]):

        model.train()

        epoch_loss = 0

        for batch_idx, (x, y) in enumerate(loader):

            x = x.to(device)
            y = y.to(device)

            optimizer.zero_grad()

            logits, _ = model(x)

            loss = criterion(
                logits.reshape(-1, processor.vocab_size),
                y.reshape(-1)
            )

            loss.backward()
            torch.nn.utils.clip_grad_norm_(
                  model.parameters(),
                  max_norm=5)
            optimizer.step()

            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 20 == 0:

                print(
                    f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
                    f"Batch [{batch_idx}/{len(loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(loader)

        losses.append(avg_loss)

        print(
            f"\nEpoch {epoch+1} Complete "
            f"| Average Loss: {avg_loss:.4f}\n"
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