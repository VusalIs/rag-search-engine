# RAG Search Engine

A search engine built from the ground up, progressively adding more sophisticated retrieval techniques — from classical keyword matching to semantic search and RAG pipelines.

## Features

- **Keyword Search** — tokenization, stopword filtering, and Porter stemming for fuzzy title matching
- **Inverted Index** — persistent index with build/save/load for fast document lookup
- **TF-IDF Scoring** — term frequency, inverse document frequency, and combined TF-IDF ranking
- **BM25** — probabilistic ranking with document length normalization and tunable k1/b parameters ([details](docs/bm25.md))
- **Semantic Search** — sentence embeddings with cosine similarity; embeddings cached to disk

## Roadmap

- [x] Keyword search
- [x] Inverted index + TF-IDF
- [x] BM25 ranking
- [x] Vector / semantic search
- [ ] Hybrid search (keyword + semantic)
- [ ] RAG pipeline

## Tech Stack

- Python 3.14
- [NLTK](https://www.nltk.org/) — NLP preprocessing
- [sentence-transformers](https://www.sbert.net/) — sentence embeddings (`all-MiniLM-L6-v2`)
- [NumPy](https://numpy.org/) — vector operations and embedding storage
- [uv](https://docs.astral.sh/uv/) — dependency management

## Setup

```bash
uv sync
source .venv/bin/activate
```

## Usage

```bash
cd cli

# Build the inverted index (run once before searching)
python keyword_search_cli.py build

# Search movies by keyword
python keyword_search_cli.py search "police"

# Term frequency of a word in a document
python keyword_search_cli.py tf <doc_id> <term>

# Inverse document frequency of a term
python keyword_search_cli.py idf <term>

# TF-IDF score for a term in a document
python keyword_search_cli.py tfidf <doc_id> <term>

# BM25 IDF for a term
python keyword_search_cli.py bm25idf <term>

# BM25 TF for a term in a document (optional k1, b params)
python keyword_search_cli.py bm25tf <doc_id> <term> [k1] [b]

# Search using full BM25 scoring
python keyword_search_cli.py bm25search "<query>"
```

```bash
cd cli

# Verify the embedding model loads correctly
python semantic_search_cli.py verify

# Embed a piece of text and inspect its dimensions
python semantic_search_cli.py embed_text "A story about loss and redemption"

# Embed a search query
python semantic_search_cli.py embed_query "action movies in space"

# Build and verify embeddings for the full movie dataset
python semantic_search_cli.py verify_embeddings

# Search movies using semantic similarity
python semantic_search_cli.py search "robots taking over the world"
python semantic_search_cli.py search "romantic comedy" --limit 3
```

## Notes

Popular indexing techniques for approximate nearest neighbor (ANN) search in vector databases:

1. **HNSW** (Hierarchical Navigable Small World) — graph-based index that builds multiple layers of proximity graphs, from coarse to fine-grained. Navigates top-down at query time for fast, high-recall ANN search. Used by FAISS, Pinecone, and Weaviate.
2. **IVF** (Inverted File Index) — partitions the vector space into clusters (Voronoi cells) using k-means. At query time, searches only the nearest cluster(s) rather than the full index, trading a small amount of recall for significant speed gains.
3. **LSH** (Locality-Sensitive Hashing) — uses hash functions designed so that similar vectors collide in the same bucket. Achieves sub-linear query time but generally lower recall than HNSW.
