# ollama_chat/file_ops.py
from pathlib import Path
from typing import Union

def read_file(path: Union[str, Path]) -> str:
    p = Path(path).expanduser()
    if not p.is_file():
        raise FileNotFoundError(f"File not found: {p}")
    return p.read_text(encoding="utf-8")


def write_file(path: Union[str, Path], content: str, overwrite: bool = False) -> None:
    p = Path(path).expanduser()
    if p.exists() and not overwrite:
        raise FileExistsError(f"File already exists: {p} (use overwrite=True)")
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(content, encoding="utf-8")
