import torch
import torch.nn as nn
from torch.utils.data import DataLoader, random_split

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

    train_size = int(0.9 * len(dataset))

    val_size = len(dataset) - train_size

    train_dataset, val_dataset = random_split(
        dataset,
        [train_size, val_size]
    )

    train_loader = DataLoader(
        train_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=True
    )

    val_loader = DataLoader(
        val_dataset,
        batch_size=CONFIG["batch_size"],
        shuffle=False
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

    train_losses = []

    val_losses = []

    perplexities = []

    print("Starting training...\n")

    for epoch in range(CONFIG["epochs"]):

        model.train()

        epoch_loss = 0

        for batch_idx, (x, y) in enumerate(train_loader):

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
                max_norm=5
            )

            optimizer.step()

            epoch_loss += loss.item()

            if batch_idx % 20 == 0:

                print(
                    f"Epoch [{epoch+1}/{CONFIG['epochs']}] "
                    f"Batch [{batch_idx}/{len(train_loader)}] "
                    f"Loss: {loss.item():.4f}"
                )

        avg_loss = epoch_loss / len(train_loader)

        model.eval()

        val_loss = 0

        with torch.no_grad():

            for x, y in val_loader:

                x = x.to(device)
                y = y.to(device)

                logits, _ = model(x)

                loss = criterion(
                    logits.reshape(-1, processor.vocab_size),
                    y.reshape(-1)
                )

                val_loss += loss.item()

        avg_val_loss = val_loss / len(val_loader)

        perplexity = torch.exp(
            torch.tensor(avg_val_loss)
        ).item()

        train_losses.append(avg_loss)
        val_losses.append(avg_val_loss)
        perplexities.append(perplexity)

        print(
            f"\nEpoch {epoch+1}/{CONFIG['epochs']} "
            f"| Train Loss: {avg_loss:.4f} "
            f"| Val Loss: {avg_val_loss:.4f} "
            f"| Perplexity: {perplexity:.2f}\n"
        )

    torch.save(
        model.state_dict(),
        "results/checkpoints/model.pth"
    )

    torch.save(
    train_losses,
    "results/checkpoints/train_losses.pt"
)
    torch.save(
    val_losses,
    "results/checkpoints/val_losses.pt"
)
    torch.save(
    perplexities,
    "results/checkpoints/perplexities.pt"
)

    print("Training complete.")


if __name__ == "__main__":
    train()