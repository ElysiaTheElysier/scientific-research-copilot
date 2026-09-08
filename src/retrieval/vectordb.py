import json
from pathlib import Path
from qdrant_client import QdrantClient
from qdrant_client.models import Distance, PointStruct, VectorParams
import torch
from sentence_transformers import SentenceTransformer

COLLECTION_NAME = "scientific_papers"
MODEL_NAME = "Qwen/Qwen3-Embedding-0.6B"
DB_PATH = "data/qdrant"


class VectorStore:
    def __init__(self, db_path: str = DB_PATH, model_name: str = MODEL_NAME, device: str = None):
        if device is None:
            device = "cuda" if torch.cuda.is_available() else "cpu"
        self.client = QdrantClient(path=db_path)
        self.model = SentenceTransformer(model_name, device=device)
        self.model.max_seq_length = 4096
        self._init_collection()

    def _init_collection(self):
        if not self.client.collection_exists(COLLECTION_NAME):
            self.client.create_collection(
                collection_name=COLLECTION_NAME,
                vectors_config=VectorParams(size=1024, distance=Distance.COSINE),
            )

    def index_chunks(self, chunks_path: str = "data/processed/chunks.json", batch_size: int = 16) -> int:
        with open(chunks_path, "r", encoding="utf-8") as f:
            chunks = json.load(f)

        texts = [c["content"] for c in chunks]
        embeddings = self.model.encode(texts, batch_size=batch_size, show_progress_bar=True)

        points = [
            PointStruct(
                id=c["chunk_id"],
                vector=emb.tolist(),
                payload=c,
            )
            for c, emb in zip(chunks, embeddings)
        ]

        for i in range(0, len(points), batch_size):
            self.client.upsert(
                collection_name=COLLECTION_NAME,
                points=points[i : i + batch_size],
            )
        return len(points)

    def search(self, query: str, limit: int = 5) -> list[dict]:
        query_vector = self.model.encode(query).tolist()
        results = self.client.query_points(
            collection_name=COLLECTION_NAME,
            query=query_vector,
            limit=limit,
        )
        return [point.payload for point in results.points]


if __name__ == "__main__":
    import sys

    store = VectorStore()
    if len(sys.argv) > 1 and sys.argv[1] == "search":
        query = " ".join(sys.argv[2:]) if len(sys.argv) > 2 else "system architecture"
        results = store.search(query)
        for r in results:
            print("=" * 60)
            print(f"[{r.get('chunk_type')}] {r.get('paper_id')} | {r.get('section')}")
            print(r.get("content")[:300])
    else:
        count = store.index_chunks()
        print(f"Indexed {count} chunks into Qdrant at {DB_PATH}")

