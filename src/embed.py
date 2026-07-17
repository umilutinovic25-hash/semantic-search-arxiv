"""Corpus loading and embedding with sentence-transformers."""

from pathlib import Path

import numpy as np
import pandas as pd
from sentence_transformers import SentenceTransformer

DATA_DIR = Path(__file__).resolve().parent.parent / "data"
CORPUS_PATH = DATA_DIR / "arxiv_ml_50k.parquet"
EMBEDDINGS_PATH = DATA_DIR / "embeddings.npy"
MODEL_NAME = "all-MiniLM-L6-v2"


def load_corpus(path: Path = CORPUS_PATH) -> pd.DataFrame:
    """Load the 50k-paper corpus (title + abstract), downloading on first use."""
    if path.exists():
        return pd.read_parquet(path)
    from datasets import load_dataset

    df = load_dataset("CShorten/ML-ArXiv-Papers", split="train").to_pandas()
    df = df[["title", "abstract"]].dropna().drop_duplicates(subset="title")
    df = df.sample(n=50_000, random_state=42).reset_index(drop=True)
    path.parent.mkdir(parents=True, exist_ok=True)
    df.to_parquet(path)
    return df


def combined_text(df: pd.DataFrame) -> list[str]:
    """Title + abstract in one string — titles carry dense signal, abstracts context."""
    return (
        df["title"].str.replace(r"\s+", " ", regex=True).str.strip()
        + ". "
        + df["abstract"].str.replace(r"\s+", " ", regex=True).str.strip()
    ).tolist()


def get_model(name: str = MODEL_NAME) -> SentenceTransformer:
    return SentenceTransformer(name)


def embed_corpus(
    df: pd.DataFrame,
    model: SentenceTransformer | None = None,
    cache: Path = EMBEDDINGS_PATH,
    batch_size: int = 256,
) -> np.ndarray:
    """Embed the corpus once and cache to disk; normalized for cosine via dot product."""
    if cache.exists():
        emb = np.load(cache)
        if len(emb) == len(df):
            return emb
    model = model or get_model()
    emb = model.encode(
        combined_text(df),
        batch_size=batch_size,
        show_progress_bar=True,
        normalize_embeddings=True,
    )
    np.save(cache, emb)
    return emb
