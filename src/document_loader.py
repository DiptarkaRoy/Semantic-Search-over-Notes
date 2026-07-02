import os
from pathlib import Path
from typing import Dict, List


def load_notes(notes_dir: str | Path) -> List[Dict[str, str]]:
    """Walks through the notes directory and extracts text from .md and .txt files.

    Returns:
        A list of dictionaries containing 'content' and 'source' path.
    """
    notes_dir = Path(notes_dir)
    documents = []

    if not notes_dir.exists():
        print(f"Warning: Directory '{notes_dir}' does not exist. Creating it.")
        notes_dir.mkdir(parents=True, exist_ok=True)
        return documents

    # Support both markdown and standard text notes
    extensions = ("*.md", "*.txt")
    file_paths = []
    for ext in extensions:
        file_paths.extend(notes_dir.glob(ext))

    for path in file_paths:
        try:
            with open(path, "r", encoding="utf-8") as f:
                content = f.read().strip()

            if content:  # Skip empty files
                documents.append(
                    {
                        "content": content,
                        "source": str(path.relative_to(notes_dir.parent.parent)),
                    }
                )
        except Exception as e:
            print(f"Error reading {path.name}: {e}")

    return documents