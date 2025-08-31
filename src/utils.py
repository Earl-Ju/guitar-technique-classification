# src/utils.py
import os

def ensure_dir(path: str) -> None:
    """
    Ensure that a directory exists.
    If it doesn't, create it (including parent dirs).
    """
    os.makedirs(path, exist_ok=True)
