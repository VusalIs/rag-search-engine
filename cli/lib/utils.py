import json
from typing import Any
from .config import DATA_PATH, GOLDEN_DATASET_PATH, SCORE_PRECISION
from .types import Movie, SearchResult, GoldenDataset


def load_movies() -> list[Movie]:
    with open(DATA_PATH, "r") as f:
        data = json.load(f)
    return data["movies"]


def load_golden_dataset() -> GoldenDataset:
    with open(GOLDEN_DATASET_PATH, "r") as f:
        return json.load(f)


def format_search_result(
    doc_id: int, title: str, document: str, score: float, **metadata: Any
) -> SearchResult:
    return {
        "id": doc_id,
        "title": title,
        "document": document,
        "score": round(score, SCORE_PRECISION),
        "metadata": metadata if metadata else {},
    }
