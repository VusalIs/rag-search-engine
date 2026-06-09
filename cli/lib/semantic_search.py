import os
import re
import numpy as np
from sentence_transformers import SentenceTransformer
from .config import CACHE_DIR, DEFAULT_CHUNK_OVERLAP, DEFAULT_CHUNK_SIZE, DEFAULT_SEARCH_LIMIT, DEFAULT_SEMANTIC_CHUNK_SIZE
from .utils import load_movies


MOVIE_EMBEDDINGS_PATH = os.path.join(CACHE_DIR, "movie_embeddings.npy")

class SemanticSearch:
    def __init__(self, model_name="all-MiniLM-L6-v2"):
        self.model = SentenceTransformer(model_name)
        self.embeddings = None
        self.documents = None
        self.document_map = {}

    def build_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        movie_strings = []
        for doc in documents:
            self.document_map[doc["id"]] = doc
            movie_strings.append(f"{doc['title']}: {doc['description']}")
        self.embeddings = self.model.encode(movie_strings, show_progress_bar=True)

        os.makedirs(os.path.dirname(MOVIE_EMBEDDINGS_PATH), exist_ok=True)
        np.save(MOVIE_EMBEDDINGS_PATH, self.embeddings)
        return self.embeddings
    
    def search(self, query, limit):
        if self.embeddings is None or self.embeddings.size == 0:
            raise ValueError(
                "No embeddings loaded. Call `load_or_create_embeddings` first."
            )

        if self.documents is None or len(self.documents) == 0:
            raise ValueError(
                "No documents loaded. Call `load_or_create_embeddings` first."
            )
        query_embedding = self.generate_embeddings(query)

        similarities = []
        for i, doc_embedding in enumerate(self.embeddings):
            similarity = cosine_similarity(query_embedding, doc_embedding)
            similarities.append((similarity, self.documents[i]))

        similarities.sort(key=lambda x: x[0], reverse=True)

        results = []
        for score, doc in similarities[:limit]:
            results.append(
                {
                    "score": score,
                    "title": doc["title"],
                    "description": doc["description"],
                }
            )

        return results

    def load_or_create_embeddings(self, documents):
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(MOVIE_EMBEDDINGS_PATH):
            self.embeddings = np.load(MOVIE_EMBEDDINGS_PATH)
            if len(self.embeddings) == len(documents):
                return self.embeddings

        return self.build_embeddings(documents)
    
    def generate_embeddings(self, text: str):
        if not text.strip():
            raise ValueError("Text is empty")
        
        encoded_text = self.model.encode([text])

        return encoded_text[0]


def verify_model() -> None:
    semantic_search = SemanticSearch()
    model = semantic_search.model

    print(f"Model loaded: {model}")
    print(f"Max sequence length: {model.max_seq_length}")

def embedded_text(text: str):
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embeddings(text)

    print(f"Text: {text}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Dimensions: {embedding.shape[0]}")

def verify_embeddings():
    search_instance = SemanticSearch()
    documents = load_movies()
    embeddings = search_instance.load_or_create_embeddings(documents)
    print(f"Number of docs:   {len(documents)}")
    print(
        f"Embeddings shape: {embeddings.shape[0]} vectors in {embeddings.shape[1]} dimensions"
    )

def cosine_similarity(vec1, vec2) -> float:
    dot_product = np.dot(vec1, vec2)
    norm1 = np.linalg.norm(vec1)
    norm2 = np.linalg.norm(vec2)

    if norm1 == 0 or norm2 == 0:
        return 0.0

    return dot_product / (norm1 * norm2)

def embed_query_text(query: str):
    semantic_search = SemanticSearch()

    embedding = semantic_search.generate_embeddings(query)

    print(f"Query: {query}")
    print(f"First 3 dimensions: {embedding[:3]}")
    print(f"Shape: {embedding.shape}")

def semantic_search(query, limit=DEFAULT_SEARCH_LIMIT):
    search_instance = SemanticSearch()
    documents = load_movies()
    search_instance.load_or_create_embeddings(documents)

    results = search_instance.search(query, limit)

    print(f"Query: {query}")
    print(f"Top {len(results)} results:")
    print()

    for i, result in enumerate(results, 1):
        print(f"{i}. {result['title']} (score: {result['score']:.4f})")
        print(f"   {result['description'][:100]}...")
        print()

def fixed_size_chunking(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> list[str]:
    words = text.split()
    chunks = []

    n_words = len(words)
    i = 0
    while i < n_words:
        chunk_words = words[i : i + chunk_size]
        chunks.append(" ".join(chunk_words))
        i = i + chunk_size - overlap

    return chunks


def chunk_text(text: str, chunk_size: int = DEFAULT_CHUNK_SIZE, overlap: int = DEFAULT_CHUNK_OVERLAP) -> None:
    chunks = fixed_size_chunking(text, chunk_size, overlap)
    print(f"Chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")

def semantic_chunk(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> list[str]:
    text = text.strip()
    if not text:
        return []
    sentences = re.split(r"(?<=[.!?])\s+", text)

    if len(sentences) == 1 and not text.endswith((".", "!", "?")):
        sentences = [text]
    
    chunks = []
    i = 0
    n_sentences = len(sentences)
    while i < n_sentences:
        chunk_sentences = sentences[i : i + max_chunk_size]
        if chunks and len(chunk_sentences) <= overlap:
            break
        cleaned_sentences = []
        for chunk_sentence in chunk_sentences:
            chunk_sentence = chunk_sentence.strip()
            if chunk_sentence:
                cleaned_sentences.append(chunk_sentence)
        if not cleaned_sentences:
            i += max_chunk_size - overlap
            continue
        chunk = " ".join(cleaned_sentences)
        chunks.append(chunk)
        i += max_chunk_size - overlap
        
    return chunks


def semantic_chunk_text(
    text: str,
    max_chunk_size: int = DEFAULT_SEMANTIC_CHUNK_SIZE,
    overlap: int = DEFAULT_CHUNK_OVERLAP,
) -> None:
    chunks = semantic_chunk(text, max_chunk_size, overlap)
    print(f"Semantically chunking {len(text)} characters")
    for i, chunk in enumerate(chunks):
        print(f"{i + 1}. {chunk}")
