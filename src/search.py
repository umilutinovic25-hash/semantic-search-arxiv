"""Cosine-similarity semantic search over the embedded corpus."""

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

from .embed import embed_corpus, get_model, load_corpus


class SemanticSearcher:
    """Embeds a natural-language query and ranks papers by cosine similarity.

    Embeddings are L2-normalized, so cosine similarity reduces to a single
    matrix-vector product over the whole corpus (exact search, no ANN index
    needed at 50k documents).
    """

    def __init__(
        self,
        df: pd.DataFrame | None = None,
        embeddings: np.ndarray | None = None,
        model: SentenceTransformer | None = None,
    ):
        self.df = df if df is not None else load_corpus()
        self.model = model or get_model()
        self.embeddings = (
            embeddings if embeddings is not None else embed_corpus(self.df, self.model)
        )

    def search(self, query: str, top_k: int = 5) -> pd.DataFrame:
        q = self.model.encode([query], normalize_embeddings=True)[0]
        scores = self.embeddings @ q
        idx = np.argsort(scores)[::-1][:top_k]
        out = self.df.iloc[idx][["title", "abstract"]].copy()
        out.insert(0, "similarity", scores[idx].round(4))
        return out.reset_index(drop=True)

    def similar_papers(self, paper_idx: int, top_k: int = 5) -> pd.DataFrame:
        """Papers most similar to a given paper (excluding itself)."""
        scores = self.embeddings @ self.embeddings[paper_idx]
        idx = np.argsort(scores)[::-1][1 : top_k + 1]
        out = self.df.iloc[idx][["title", "abstract"]].copy()
        out.insert(0, "similarity", scores[idx].round(4))
        return out.reset_index(drop=True)


def main():
    import argparse

    parser = argparse.ArgumentParser(description="Semantic search over ArXiv ML papers")
    parser.add_argument("query", help="natural-language query")
    parser.add_argument("-k", "--top-k", type=int, default=5)
    args = parser.parse_args()

    results = SemanticSearcher().search(args.query, args.top_k)
    for _, row in results.iterrows():
        print(f"[{row['similarity']:.3f}] {' '.join(row['title'].split())}")


if __name__ == "__main__":
    main()
