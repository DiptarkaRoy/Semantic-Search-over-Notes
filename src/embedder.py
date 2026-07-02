import torch
from sentence_transformers import SentenceTransformer


class NotesEmbedder:

    def __init__(self, model_name: str = "all-MiniLM-L6-v2"):
        """Initializes the embedding model and selects the best available hardware."""
        # Check for Apple Silicon GPU acceleration (MPS)
        if torch.backends.mps.is_available():
            self.device = "mps"
        elif torch.cuda.is_available():
            self.device = "cuda"
        else:
            self.device = "cpu"

        print(f"Loading embedding model '{model_name}' on device: {self.device}")
        self.model = SentenceTransformer(model_name, device=self.device)

    def compute_embeddings(self, texts: list[str]) -> list:
        """Converts a list of text strings into their vector representations."""
        if not texts:
            return []
        # show_progress_bar is built directly into sentence-transformers encoding
        return self.model.encode(texts, show_progress_bar=True)

    def compute_query_embedding(self, query: str):
        """Converts a single search query string into a vector."""
        return self.model.encode(query, show_progress_bar=False)