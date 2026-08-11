import pickle
from abc import ABC, abstractmethod
from pathlib import Path

import faiss
import numpy as np

from config import (
    EMBEDDINGS_FILE,
    FAISS_INDEX_FILE,
    QDRANT_API_KEY,
    QDRANT_COLLECTION,
    QDRANT_URL,
    VECTOR_BACKEND,
)


class VectorBackend(ABC):
    @abstractmethod
    def is_ready(self) -> bool:
        ...

    @abstractmethod
    def build(self, chunks: list, vectors: np.ndarray) -> None:
        ...

    @abstractmethod
    def search(self, query_vector: np.ndarray, k: int) -> list:
        ...


class FaissBackend(VectorBackend):
    """Local/dev backend: index persisted as a flat file, chunk lookup by position."""

    def __init__(self, chunks: list):
        self.chunks = chunks
        self.index_path = Path(FAISS_INDEX_FILE)
        self.index = faiss.read_index(str(self.index_path)) if self.index_path.exists() else None

    def is_ready(self) -> bool:
        return self.index is not None

    def build(self, chunks: list, vectors: np.ndarray) -> None:
        index = faiss.IndexFlatL2(vectors.shape[1])
        index.add(vectors)

        self.index_path.parent.mkdir(parents=True, exist_ok=True)
        faiss.write_index(index, str(self.index_path))
        with open(EMBEDDINGS_FILE, "wb") as f:
            pickle.dump(vectors, f)

        self.index = index
        self.chunks = chunks

    def search(self, query_vector: np.ndarray, k: int) -> list:
        if not self.is_ready():
            raise RuntimeError("FAISS index belum tersedia. Jalankan build_index.py terlebih dahulu.")
        _, idxs = self.index.search(np.array([query_vector]), k)
        return [self.chunks[i] for i in idxs[0] if i != -1]


class QdrantBackend(VectorBackend):
    """Staging/production backend: chunk text/source stored as point payload, so
    search results are self-contained and don't depend on chunks.json ordering."""

    def __init__(self, chunks: list):
        from qdrant_client import QdrantClient

        if not QDRANT_URL:
            raise ValueError("QDRANT_URL belum di-set di env/.env untuk VECTOR_BACKEND=qdrant")

        self.chunks = chunks
        self.collection_name = QDRANT_COLLECTION
        self.client = QdrantClient(location=QDRANT_URL, api_key=QDRANT_API_KEY)

    def is_ready(self) -> bool:
        return self.client.collection_exists(self.collection_name)

    def build(self, chunks: list, vectors: np.ndarray) -> None:
        from qdrant_client.models import Distance, PointStruct, VectorParams

        self.client.recreate_collection(
            collection_name=self.collection_name,
            vectors_config=VectorParams(size=vectors.shape[1], distance=Distance.COSINE),
        )
        points = [
            PointStruct(id=i, vector=vectors[i].tolist(), payload=chunks[i])
            for i in range(len(chunks))
        ]
        self.client.upload_points(collection_name=self.collection_name, points=points, wait=True)
        self.chunks = chunks

    def search(self, query_vector: np.ndarray, k: int) -> list:
        if not self.is_ready():
            raise RuntimeError(
                f"Koleksi Qdrant '{self.collection_name}' belum tersedia. Jalankan build_index.py terlebih dahulu."
            )
        hits = self.client.query_points(
            collection_name=self.collection_name,
            query=query_vector.tolist(),
            limit=k,
        ).points
        return [hit.payload for hit in hits]


def get_backend(chunks: list) -> VectorBackend:
    if VECTOR_BACKEND == "qdrant":
        return QdrantBackend(chunks)
    if VECTOR_BACKEND == "faiss":
        return FaissBackend(chunks)
    raise ValueError(f"VECTOR_BACKEND tidak dikenal: '{VECTOR_BACKEND}' (gunakan 'faiss' atau 'qdrant')")
