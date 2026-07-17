"""Interactive semantic-search demo over 50k ArXiv ML papers.

Type a query in your own words and compare, side by side, what
embedding-based semantic search finds versus classic TF-IDF keyword
search — plus a "more like this" tab powered by the same embeddings.

Run:  python app.py   ->  http://127.0.0.1:7862
"""

import time

import gradio as gr
import numpy as np
import pandas as pd
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.metrics.pairwise import cosine_similarity

from src.embed import combined_text, embed_corpus, get_model, load_corpus

print("loading corpus and embeddings...")
DF = load_corpus()
MODEL = get_model()
EMB = embed_corpus(DF, MODEL)
TITLES = DF["title"].str.replace(r"\s+", " ", regex=True).str.strip()

print("fitting TF-IDF baseline (once, ~10s)...")
TEXTS = combined_text(DF)
TFIDF = TfidfVectorizer(stop_words="english", max_features=50_000)
TFIDF_MATRIX = TFIDF.fit_transform(TEXTS)

EXAMPLES = [
    "teaching robots to walk using trial and error",
    "how can I make a neural network smaller and faster for my phone",
    "detecting fake accounts and bots on social networks",
    "predicting which customers will stop using a service",
    "generating realistic images from text descriptions",
]

EXPLAINER = """
### What is this?

A search engine over **50,000 machine-learning papers** from ArXiv that understands
*meaning*, not just words. Each paper (title + abstract) is turned into a
384-dimensional vector by a sentence-transformer model; your query becomes a vector
the same way, and papers are ranked by cosine similarity — one matrix product over
the whole corpus, a few milliseconds per query, no vector database needed at this scale.

### Why the side-by-side comparison?

The right column shows what classic **keyword search (TF-IDF)** returns for the same
query. Try *"teaching robots to walk using trial and error"*: the semantic column finds
reinforcement-learning locomotion papers even though the query never says
"reinforcement learning" — the keyword column latches onto the literal words
("teaching", "walk", "error") and misses the concept. That gap is the whole point
of embeddings.

### More like this

The same vectors power paper-to-paper recommendations: pick any paper and get its
nearest neighbours in embedding space — no extra model, no extra index.
"""


def _fmt(titles, scores, abstracts=None, limit_abs=180):
    rows = {"score": [f"{s:.3f}" for s in scores], "title": list(titles)}
    if abstracts is not None:
        rows["abstract (start)"] = [
            " ".join(a.split())[:limit_abs] + "…" for a in abstracts
        ]
    return pd.DataFrame(rows)


def search_both(query: str, top_k: int):
    query = (query or "").strip()
    if not query:
        empty = pd.DataFrame()
        return "Type a query first. 👆", empty, empty

    top_k = int(top_k)

    t0 = time.perf_counter()
    q = MODEL.encode([query], normalize_embeddings=True)[0]
    scores = EMB @ q
    idx = np.argsort(scores)[::-1][:top_k]
    sem_ms = (time.perf_counter() - t0) * 1000
    sem_df = _fmt(TITLES.iloc[idx], scores[idx], DF["abstract"].iloc[idx])

    qk = TFIDF.transform([query])
    kscores = cosine_similarity(qk, TFIDF_MATRIX)[0]
    kidx = np.argsort(kscores)[::-1][:top_k]
    kw_df = _fmt(TITLES.iloc[kidx], kscores[kidx])

    overlap = len(set(idx) & set(kidx))
    info = (
        f"**{len(DF):,} papers searched in {sem_ms:.0f} ms** (encode query + one matrix product). "
        f"Overlap between the two result lists: **{overlap}/{top_k}** — "
        + ("the methods largely agree on this query."
           if overlap >= top_k // 2 else
           "the methods disagree; check which column actually answers your question.")
    )
    return info, sem_df, kw_df


def find_titles(fragment: str):
    fragment = (fragment or "").strip()
    if len(fragment) < 3:
        return gr.Dropdown(choices=[], value=None)
    mask = TITLES.str.contains(fragment, case=False, regex=False)
    hits = TITLES[mask].head(20).tolist()
    return gr.Dropdown(choices=hits, value=hits[0] if hits else None)


def more_like_this(title: str, top_k: int):
    if not title:
        return "Pick a paper from the dropdown first. 👆", pd.DataFrame()
    matches = np.where(TITLES == title)[0]
    if len(matches) == 0:
        return "Paper not found — search for it again.", pd.DataFrame()
    i = int(matches[0])
    scores = EMB @ EMB[i]
    idx = np.argsort(scores)[::-1][1 : int(top_k) + 1]
    out = _fmt(TITLES.iloc[idx], scores[idx], DF["abstract"].iloc[idx])
    return f"Papers closest to **{title}** in embedding space:", out


with gr.Blocks(title="Semantic Search — ArXiv ML Papers") as demo:
    gr.Markdown(
        "# 🔎 Semantic Search over 50,000 ArXiv ML Papers\n"
        "Ask in your own words — the engine matches **meaning**, not keywords. "
        "The right column shows what plain keyword search would have found instead."
    )
    with gr.Accordion("ℹ️ How does this work? (click)", open=False):
        gr.Markdown(EXPLAINER)

    with gr.Tabs():
        with gr.Tab("🔍 Search"):
            with gr.Row():
                query = gr.Textbox(label="Your question or topic, in plain language",
                                   placeholder=EXAMPLES[0], scale=4)
                top_k = gr.Slider(3, 15, value=5, step=1, label="Results", scale=1)
            gr.Examples(examples=[[e] for e in EXAMPLES], inputs=[query], label="Try these")
            btn = gr.Button("🔎 Search", variant="primary", size="lg")
            info = gr.Markdown("")
            with gr.Row():
                sem_df = gr.Dataframe(label="🧠 Semantic search (embeddings + cosine)", interactive=False)
                kw_df = gr.Dataframe(label="🔤 Keyword search (TF-IDF baseline)", interactive=False)
            btn.click(search_both, inputs=[query, top_k], outputs=[info, sem_df, kw_df])
            query.submit(search_both, inputs=[query, top_k], outputs=[info, sem_df, kw_df])

        with gr.Tab("📄 More like this"):
            gr.Markdown("Find a paper by a fragment of its title, then get its nearest neighbours "
                        "in embedding space — an instant related-work list.")
            with gr.Row():
                frag = gr.Textbox(label="Title fragment (e.g. 'attention is all')", scale=3)
                pick = gr.Dropdown(label="Matching papers", choices=[], scale=4)
                k2 = gr.Slider(3, 15, value=5, step=1, label="Neighbours", scale=1)
            btn2 = gr.Button("📄 Find related papers", variant="primary", size="lg")
            info2 = gr.Markdown("")
            rel_df = gr.Dataframe(label="Nearest neighbours", interactive=False)
            frag.change(find_titles, inputs=frag, outputs=pick)
            btn2.click(more_like_this, inputs=[pick, k2], outputs=[info2, rel_df])

if __name__ == "__main__":
    demo.launch(
        server_name="127.0.0.1", server_port=7862, inbrowser=False,
        theme=gr.themes.Soft(primary_hue="blue", neutral_hue="slate"),
    )
