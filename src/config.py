"""Load local secrets without overriding the caller's environment."""
from pathlib import Path
from dotenv import load_dotenv

PROJECT_ROOT = Path(__file__).resolve().parent.parent


def load_config():
    load_dotenv(PROJECT_ROOT / ".env", override=False)
