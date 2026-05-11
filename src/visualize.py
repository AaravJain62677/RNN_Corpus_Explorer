import torch
import matplotlib.pyplot as plt


losses = torch.load(
    "results/checkpoints/losses.pt"
)

plt.plot(losses)

plt.xlabel("Epoch")

plt.ylabel("Loss")

plt.title("Training Loss")

plt.savefig(
    "results/plots/loss_curve.png"
)

plt.show()