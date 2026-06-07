# RAG Search Engine

A search engine built from the ground up, progressively adding more sophisticated retrieval techniques — from classical keyword matching to semantic search and RAG pipelines.

## Features

- **Keyword Search** — tokenization, stopword filtering, and Porter stemming for fuzzy title matching
- **Inverted Index** — persistent index with build/save/load for fast document lookup
- **TF-IDF Scoring** — term frequency, inverse document frequency, and combined TF-IDF ranking

## Roadmap

- [x] Keyword search
- [x] Inverted index + TF-IDF
- [ ] BM25 ranking
- [ ] Vector / semantic search
- [ ] Hybrid search (keyword + semantic)
- [ ] RAG pipeline

## Tech Stack

- Python 3.14
- [NLTK](https://www.nltk.org/) — NLP preprocessing
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
```
