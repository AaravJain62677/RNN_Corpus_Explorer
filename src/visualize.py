import torch
import matplotlib.pyplot as plt


train_losses = torch.load(
    "results/checkpoints/train_losses.pt"
)

val_losses = torch.load(
    "results/checkpoints/val_losses.pt"
)

perplexities = torch.load(
    "results/checkpoints/perplexities.pt"
)

# Training vs Validation Loss

plt.figure(figsize=(8, 5))

plt.plot(
    train_losses,
    marker="o",
    label="Train Loss"
)

plt.plot(
    val_losses,
    marker="o",
    label="Validation Loss"
)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training vs Validation Loss")

plt.grid(True)

plt.legend()

plt.savefig(
    "results/plots/loss_curve.png"
)

plt.close()

# Perplexity Curve

plt.figure(figsize=(8, 5))

plt.plot(
    perplexities,
    marker="o"
)

plt.xlabel("Epoch")

plt.ylabel("Perplexity")

plt.title("Perplexity Curve")

plt.grid(True)

plt.savefig(
    "results/plots/perplexity_curve.png"
)

plt.close()


print("Plots saved successfully.")