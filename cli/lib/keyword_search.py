import os
import math
import pickle
import string
from collections import defaultdict, Counter

from nltk.stem import PorterStemmer

from .search_utils import (
    CACHE_DIR,
    DEFAULT_SEARCH_LIMIT,
    STOPWORDS_PATH,
    load_movies,
)


class InvertedIndex:
    def __init__(self) -> None:
        self.index: defaultdict[str, set] = defaultdict(set)
        self.docmap: dict[int, dict] = {}
        self.term_frequencies: defaultdict[int, Counter] = defaultdict(Counter)

        self.index_path = os.path.join(CACHE_DIR, "index.pkl")
        self.docmap_path = os.path.join(CACHE_DIR, "docmap.pkl")
        self.term_frequencies_path = os.path.join(CACHE_DIR, "term_frequencies.pkl")

    def build(self) -> None:
        movies = load_movies()
        for m in movies:
            doc_id = m["id"]
            doc_description = f"{m['title']} {m['description']}"
            self.docmap[doc_id] = m
            self.__add_document(doc_id, doc_description)

    def save(self) -> None:
        os.makedirs(CACHE_DIR, exist_ok=True)
        with open(self.index_path, "wb") as f:
            pickle.dump(self.index, f)
        with open(self.docmap_path, "wb") as f:
            pickle.dump(self.docmap, f)
        with open(self.term_frequencies_path, "wb") as f:
            pickle.dump(self.term_frequencies, f)

    def load(self) -> None:
        with open(self.index_path, "rb") as f:
            self.index = pickle.load(f) 
        with open(self.docmap_path, "rb") as f:
            self.docmap = pickle.load(f)
        with open(self.term_frequencies_path, "rb") as f:
            self.term_frequencies = pickle.load(f)

    def get_documents(self, term: str) -> list[int]:
        doc_ids = self.index.get(term, set())
        return sorted(list(doc_ids))

    def get_tf(self, doc_id: int, token: str) -> int:
        return self.term_frequencies[doc_id][token]
    
    def get_idf(self, term: str) -> float:
        doc_count = len(self.docmap)
        term_doc_count = len(self.index[term])
        return math.log((doc_count + 1) / (term_doc_count + 1))
    
    def get_tf_idf(self, doc_id: int, term: str) -> int:
        return self.get_tf(doc_id, term) * self.get_idf(term)


    def __add_document(self, doc_id: int, text: str) -> None:
        tokens = tokenize_text(text)

        for token in set(tokens):
            self.index[token].add(doc_id)

        self.term_frequencies[doc_id].update(tokens)



def build_command() -> None:
    idx = InvertedIndex()
    idx.build()
    idx.save()

def search_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> list[dict]:
    idx = InvertedIndex()
    idx.load()
    results = []
    seen = set()
    tokenized_queries = tokenize_text(query)

    for tq in tokenized_queries:
        documents_ids = idx.get_documents(tq)

        for did in documents_ids:
            if did in seen:
                continue
            seen.add(did)
            results.append(idx.docmap[did])
            if len(results) >= limit:
                return results
            
    return results

def tf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    token = tokenize_single_term(term)

    return idx.get_tf(doc_id, token)

def idf_command(term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    token = tokenize_single_term(term)

    return idx.get_idf(token)

def tf_idf_command(doc_id: int, term: str) -> int:
    idx = InvertedIndex()
    idx.load()
    
    return idx.get_tf_idf(doc_id, tokenize_single_term(term))


def preprocess_text(text: str) -> list[str]:
    text = text.lower()
    text = text.translate(str.maketrans("", "", string.punctuation))
    return text

def load_stopwords() -> list[str]:
    with open(STOPWORDS_PATH, "r") as f:
        return [preprocess_text(word) for word in f.read().splitlines()]
    
STOPWORDS = load_stopwords()
stemmer = PorterStemmer()

def tokenize_single_term(term: str) -> str:
    tokens = tokenize_text(term)
    if len(tokens) != 1:
        raise ValueError("term must be a single token")
    return tokens[0]

def tokenize_text(text: str) -> list[str]:
    text = preprocess_text(text)
    tokens = text.split()
    valid_tokens = []

    for token in tokens:
        if token:
            valid_tokens.append(token)

    filtered_words = []

    for word in valid_tokens:
        if word not in STOPWORDS:
            filtered_words.append(word)

    stemmed_words = []
    for word in filtered_words:
        stemmed_words.append(stemmer.stem(word))

    return stemmed_words