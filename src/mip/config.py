import os
from dataclasses import dataclass
from pathlib import Path
# from dotenv import load_dotenv

# .env load
# load_dotenv()

@dataclass(frozen=True)
class Config:
    PROJECT_ROOT: Path = Path(__file__).resolve().parents[2]
    DATA_DIR: Path = PROJECT_ROOT / "data"
    FIG_DIR: Path = PROJECT_ROOT / "figures"
    SAVED_MODELS_DIR: Path = PROJECT_ROOT / "saved_models"
    SAVED_MU_DIR: Path = PROJECT_ROOT / "saved_mu"
    SEED: int = 42
    TEST_SIZE: float = 0.2

config = Config()