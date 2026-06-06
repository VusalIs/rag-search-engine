# RAG Search Engine

A search engine built from the ground up, progressively adding more sophisticated retrieval techniques — from classical keyword matching to semantic search and RAG pipelines.

## Features

- **Keyword Search** — tokenization, stopword filtering, and Porter stemming for fuzzy title matching

## Roadmap

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
python keyword_search_cli.py search "<query>"
```

```bash
python keyword_search_cli.py search "police"
# 1. Kaakha..Kaakha: The Police

python keyword_search_cli.py search "love story"
# 1. Love Story
# 2. Love Story 2050
```
