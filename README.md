# Semantic Search in Research Papers

Semantic search over **50,000 ArXiv machine-learning papers** using
sentence-transformer embeddings and cosine similarity — finding papers by
*meaning*, not keywords. Includes a side-by-side comparison against TF-IDF
keyword search, related-paper recommendations and an embedding-space
visualization.

## Highlights

- **Title + abstract fusion**: each paper is embedded as one dense 384-dim vector
  (`all-MiniLM-L6-v2`), capturing deeper semantic context than either field alone.
- **Exact cosine search**: embeddings are L2-normalized, so ranking all 50k papers
  is a single matrix-vector product — milliseconds per query, no vector DB needed.
- **Semantic vs. keyword**: a query like *"teaching robots to walk using trial and
  error"* retrieves reinforcement-learning locomotion papers with zero term
  overlap — TF-IDF can't.
- **More-like-this**: nearest neighbours in embedding space double as a
  related-paper recommender.
- **Embedding-space map**: t-SNE + KMeans over the corpus reveals coherent
  research communities, auto-labeled by their most distinctive title terms.

## Usage

```bash
python -m venv .venv && source .venv/bin/activate
pip install -r requirements.txt

# CLI search (downloads the corpus + builds the embedding cache on first run)
python -m src.search "how to detect fraudulent credit card transactions"

# full analysis
jupyter notebook notebooks/semantic_search.ipynb
```

## Project structure

```
├── notebooks/
│   └── semantic_search.ipynb   # full analysis, executed with outputs
├── src/
│   ├── embed.py                # corpus loading, title+abstract fusion, embedding cache
│   └── search.py               # SemanticSearcher (cosine ranking) + CLI
├── data/                       # corpus + embedding cache (auto-built, not committed)
└── requirements.txt
```

## Dataset

[ML-ArXiv-Papers](https://huggingface.co/datasets/CShorten/ML-ArXiv-Papers) —
titles and abstracts of machine-learning papers from ArXiv; a 50k sample is
drawn deterministically (seed 42) on first run.
