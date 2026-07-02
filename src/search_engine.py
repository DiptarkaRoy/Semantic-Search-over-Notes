import pickle
from pathlib import Path
from typing import Dict, List, Any
import numpy as np

from src.document_loader import load_notes
from src.embedder import NotesEmbedder


class SearchEngine:
    def __init__(self, data_dir: str | Path = "data"):
        self.data_dir = Path(data_dir)
        self.notes_dir = self.data_dir / "notes"
        self.cache_path = self.data_dir / "index.pkl"
        
        self.embedder = NotesEmbedder()
        self.documents: List[Dict[str, str]] = []
        self.embeddings: np.ndarray = np.empty((0, 0))

    def build_or_load_index(self, force_reindex: bool = False):
        """Loads cached embeddings if they exist, otherwise reads documents and encodes them."""
        # 1. Load raw markdown documents
        self.documents = load_notes(self.notes_dir)
        if not self.documents:
            print("No notes found to index. Drop some files into data/notes/")
            return

        # 2. Check cache unless forced to reindex
        if not force_reindex and self.cache_path.exists():
            print(f"Loading cached embeddings from {self.cache_path}...")
            try:
                with open(self.cache_path, "rb") as f:
                    cache_data = pickle.load(f)
                
                # Check if document count matches cache to prevent stale indices
                if len(cache_data.get("documents", [])) == len(self.documents):
                    self.documents = cache_data["documents"]
                    self.embeddings = cache_data["embeddings"]
                    print("Cache loaded successfully.")
                    return
                else:
                    print("Changes detected in notes directory. Re-indexing...")
            except Exception as e:
                print(f"Failed to read cache ({e}). Re-indexing...")

        # 3. Compute new embeddings if cache missed or stale
        print(f"Indexing {len(self.documents)} documents...")
        texts = [doc["content"] for doc in self.documents]
        self.embeddings = np.array(self.embedder.compute_embeddings(texts))

        # 4. Save to cache
        try:
            self.data_dir.mkdir(parents=True, exist_ok=True)
            with open(self.cache_path, "wb") as f:
                pickle.dump({"documents": self.documents, "embeddings": self.embeddings}, f)
            print(f"Embeddings cached to {self.cache_path}")
        except Exception as e:
            print(f"Failed to write cache: {e}")

    def search(self, query: str, top_k: int = 3) -> List[Dict[str, Any]]:
        """Performs cosine similarity search between query and indexed notes."""
        if self.embeddings.size == 0 or not self.documents:
            print("Search index is empty. Please index documents first.")
            return []

        # Generate query vector
        query_vector = self.embedder.compute_query_embedding(query)

        # Vectorized Cosine Similarity Math
        # cos_sim = (A . B) / (||A|| * ||B||)
        dot_product = np.dot(self.embeddings, query_vector)
        matrix_norms = np.linalg.norm(self.embeddings, axis=1)
        query_norm = np.linalg.norm(query_vector)
        
        # Avoid division by zero warnings
        scores = dot_product / (matrix_norms * query_norm + 1e-8)

        # Get top-k indices sorted descending
        top_indices = np.argsort(scores)[::-1][:top_k]

        results = []
        for idx in top_indices:
            results.append({
                "score": float(scores[idx]),
                "content": self.documents[idx]["content"],
                "source": self.documents[idx]["source"]
            })
            
        return results