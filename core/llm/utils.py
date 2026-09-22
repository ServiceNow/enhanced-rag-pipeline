import yaml
from pathlib import Path

import torch
from utils.utils import LOGGER

MODEL_INVENTORY_PATH = "model_inventory.yaml"


def get_compute_dtype() -> torch.dtype:
    # Check GPU compatibility with bfloat16
    if not torch.cuda.is_available():
        return torch.float32
    major, _ = torch.cuda.get_device_capability()
    if major >= 8:
        LOGGER.info("=" * 80 + "\nThe GPU supports bfloat16. Using it. \n" + "=" * 80)
        compute_dtype = torch.bfloat16
    else:
        compute_dtype = torch.float16
    return compute_dtype


def get_model_from_name(model_name_path: str) -> str:
    model_card_path = Path(__file__).resolve().parent / MODEL_INVENTORY_PATH
    with open(model_card_path, "r") as f:
        model_cards: dict = yaml.safe_load(f)

    return model_cards.get(model_name_path, model_name_path)
