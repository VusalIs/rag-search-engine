from ..config import BM25_B, DEFAULT_SEARCH_LIMIT
from ..index import InvertedIndex
from ..preprocessing import tokenize_single_term
from ..types import SearchResult


def bm25_idf_command(term: str) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_idf(tokenize_single_term(term))


def bm25_tf_command(doc_id: int, term: str, k1: float, b: float = BM25_B) -> float:
    idx = InvertedIndex()
    idx.load()
    return idx.get_bm25_tf(doc_id, tokenize_single_term(term), k1, b)


def bm25search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[SearchResult]:
    idx = InvertedIndex()
    idx.load()
    return idx.bm25_search(query, limit)
