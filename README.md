# Local Notes Semantic Search Engine

A lightweight, local semantic search engine built to conceptually query your personal markdown and text notes. Instead of relying on exact keyword matching, this tool uses pre-trained Transformer embeddings to match the underlying *meaning* of your search terms.

---

## 🛠️ Tech Stack & Features
* **Core Framework:** `sentence-transformers` utilizing the highly efficient `all-MiniLM-L6-v2` model.
* **Vector Math:** Vectorized cosine similarity operations managed via `numpy`.
* **Hardware Acceleration:** Native Apple Silicon (`mps`) GPU acceleration support.
* **Package Management:** Fast environment execution powered by `uv`.
* **Caching:** Local persistence of computed vector embeddings via `pickle` to bypass redundant model passes.

---

## Project Structure

```text
semantic_notes_search/
│
├── data/
│   ├── notes/               # Place your local .md and .txt notes here
│   └── index.pkl            # Local binary cache for computed vector embeddings
│
├── src/
│   ├── __init__.py          # Marks src as a package and cleans up imports
│   ├── document_loader.py   # Scans directory and parses file data
│   ├── embedder.py          # Manages model loads and vector translations
│   └── search_engine.py     # Coordinates indexing logic and vector matrix operations
│
├── app.py                   # Main terminal CLI interface
└── pyproject.toml           # Declarative uv project dependencies
```

## Getting Started
* **Prerequisites:** Ensure you have `uv` installed in you machine. If not, run the following:
```bash
brew install uv
```
* **Add Your Notes:** Create the storage folder structure and populate it with some markdown (`*.md`) or text (`*.txt`) notes:
```bash
mkdir -p data/notes
```
* **Run the Search Engine:** Execute the application using uv. It will automatically spin up an isolated virtual environment, install the declarative dependencies, and load the text interface:
```bash
uv run app.py
```
## Commands
Once the interface boots up, you can execute standard conversational or conceptual queries right inside the prompt:
* **Query Search**: Type your conceptual phrase (e.g., neural networks, high protein diet) and press Enter.
* **:reindex**– Force the script to re-scan the file path and recompute embeddings if you modified or added notes.
* **:exit** – Terminate the application cleanly.