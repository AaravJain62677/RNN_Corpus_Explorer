import torch
import torch.nn.functional as F

from src.config import CONFIG
from src.preprocess import TextProcessor
from src.model import RNNModel


def generate_text(
    start_text,
    length=300,
    temperature=1.0
):

    device = torch.device(
        CONFIG["device"]
        if torch.cuda.is_available()
        else "cpu"
    )

    print("Loading processor...")

    processor = TextProcessor(
        CONFIG["dataset_path"]
    )

    print("Loading model...")

    model = RNNModel(
        vocab_size=processor.vocab_size,
        embedding_dim=CONFIG["embedding_dim"],
        hidden_size=CONFIG["hidden_size"],
        num_layers=CONFIG["num_layers"],
        model_type=CONFIG["model_type"]
    ).to(device)

    model.load_state_dict(
        torch.load(
            "results/checkpoints/model.pth",
            map_location=device
        )
    )

    model.eval()

    print("Generating text...\n")

    input_indices = processor.encode(start_text)

    input_tensor = torch.tensor(
        input_indices,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    generated = start_text

    hidden = None

    for _ in range(length):

        with torch.no_grad():

            logits, hidden = model(
                input_tensor,
                hidden
            )

        logits = logits[:, -1, :] / temperature

        probs = F.softmax(logits, dim=-1)

        next_idx = torch.multinomial(
            probs,
            num_samples=1
        ).item()

        next_char = processor.idx_to_char[next_idx]

        generated += next_char

        input_tensor = torch.tensor(
            [[next_idx]],
            dtype=torch.long
        ).to(device)

    return generated


if __name__ == "__main__":

    text = generate_text(
        start_text="The ",
        length=500,
        temperature=0.8
    )

    print(text)