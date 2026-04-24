
import sys
from pathlib import Path


def get_base_dir() -> Path:
    if getattr(sys, 'frozen', False):
        return Path(sys._MEIPASS)
    return Path(__file__).resolve().parent.parent.parent


BASE_DIR: Path = get_base_dir()

