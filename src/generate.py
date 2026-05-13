import torch
import torch.nn.functional as F

from src.config import CONFIG
from src.preprocess import TextProcessor
from src.model import RNNModel
from src.utils import get_device


def generate_text(
    start_text,
    length=300,
    temperature=1.0
):

    device = get_device()

    processor = TextProcessor(
        CONFIG["dataset_path"]
    )

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

    input_indices = processor.encode(start_text)

    input_tensor = torch.tensor(
        input_indices,
        dtype=torch.long
    ).unsqueeze(0).to(device)

    generated = start_text
    print(start_text)