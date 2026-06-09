

import json
import os

import numpy as np

from cli.lib.config import CHUNK_EMBEDDINGS_PATH, CHUNK_METADATA_PATH, DEFAULT_SEARCH_LIMIT, DOCUMENT_PREVIEW_LENGTH
from cli.lib.semantic_search import SemanticSearch, cosine_similarity, semantic_chunk
from cli.lib.utils import format_search_result, load_movies


class ChunkedSemanticSearch(SemanticSearch):
    def __init__(self, model_name: str = "all-MiniLM-L6-v2") -> None:
         super().__init__(model_name)
         self.chunk_embeddings = None
         self.chunk_metadata = None

    def build_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents

        for doc in documents:
            self.document_map[doc["id"]] = doc

        all_chunks: list[str] = []
        chunk_metadata: list[dict] = []

        for idx, doc in enumerate(documents):
            text = doc.get("description", "")
            if not text.strip():
                continue

            chunks = semantic_chunk(text, 4, 1)
            
            for i, chunk in enumerate(chunks):
                all_chunks.append(chunk)
                chunk_metadata.append(
                    {"movie_idx": idx, "chunk_idx": i, "total_chunks": len(chunks)}
                )

        self.chunk_embeddings = self.model.encode(all_chunks)
        self.chunk_metadata = chunk_metadata

        os.makedirs(os.path.dirname(CHUNK_EMBEDDINGS_PATH), exist_ok=True)
        np.save(CHUNK_EMBEDDINGS_PATH, self.chunk_embeddings)
        with open(CHUNK_METADATA_PATH, "w") as f:
            json.dump(
                {"chunks": chunk_metadata, "total_chunks": len(all_chunks)}, f, indent=2
            )

        return self.chunk_embeddings
    
    def load_or_create_chunk_embeddings(self, documents: list[dict]) -> np.ndarray:
        self.documents = documents
        self.document_map = {}
        for doc in documents:
            self.document_map[doc["id"]] = doc

        if os.path.exists(CHUNK_EMBEDDINGS_PATH) and os.path.exists(
            CHUNK_METADATA_PATH
        ):
            self.chunk_embeddings = np.load(CHUNK_EMBEDDINGS_PATH)
            with open(CHUNK_METADATA_PATH, "r") as f:
                data = json.load(f)
                self.chunk_metadata = data["chunks"]
            return self.chunk_embeddings

        return self.build_chunk_embeddings(documents)
    
    def search_chunks(self, query: str, limit: int = 10):
        if self.chunk_embeddings is None or self.chunk_metadata is None:
            raise ValueError(
                "No chunk embeddings loaded. Call load_or_create_chunk_embeddings first."
            )
        
        query_embedding = self.generate_embeddings(query)

        chunk_scores: list[dict]= []

        for idx, chunk_embedding in enumerate(self.chunk_embeddings):
            cos_sim = cosine_similarity(chunk_embedding, query_embedding)
            chunk_scores.append({
                "chunk_idx": self.chunk_metadata[idx]["chunk_idx"],
                "movie_idx": self.chunk_metadata[idx]["movie_idx"],
                "score": cos_sim
            })

        movie_scores = {}
        for chunk_score in chunk_scores:
            movie_idx = chunk_score["movie_idx"]
            if (
                movie_idx not in movie_scores
                or chunk_score["score"] > movie_scores[movie_idx]
            ):
                movie_scores[movie_idx] = chunk_score["score"]

        sorted_movies = sorted(movie_scores.items(), key=lambda x: x[1], reverse=True)

        results = []
        for movie_idx, score in sorted_movies[:limit]:
            if movie_idx is None:
                continue
            doc = self.documents[movie_idx]
            results.append(
                format_search_result(
                    doc_id=doc["id"],
                    title=doc["title"],
                    document=doc["description"][:DOCUMENT_PREVIEW_LENGTH],
                    score=score,
                )
            )

        return results
    

def search_chunked_command(query: str, limit: int = DEFAULT_SEARCH_LIMIT) -> dict:
    movies = load_movies()
    searcher = ChunkedSemanticSearch()
    searcher.load_or_create_chunk_embeddings(movies)
    results = searcher.search_chunks(query, limit)
    return {"query": query, "results": results}


def embed_chunks_command() -> np.ndarray:
    movies = load_movies()
    searcher = ChunkedSemanticSearch()
    return searcher.load_or_create_chunk_embeddings(movies)