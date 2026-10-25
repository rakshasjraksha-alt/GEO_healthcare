"""
================================================================================
indexing.py - Dual-Collection ChromaDB Indexing with Sentence-Transformers
================================================================================
Department of Computer Science & Engineering (AI & ML)
Siddaganga Institute of Technology, Tumakuru
Mini-Project (Phase 1/2) - V Semester

Module Overview:
----------------
This module builds and manages the local persistent vector database (ChromaDB)
for our privacy-first Generative Engine Optimization (GEO) framework.

Architecture:
1. Embedding Model:
   - sentence-transformers/all-MiniLM-L6-v2
   - Produces 384-dimensional dense semantic vectors.
   - Highly optimized for CPU inference on standard laptops (Core i5 / Ryzen 5).
   - Uses Cosine Similarity space for normalized distance comparisons.
2. Dual Persistent Collections:
   - Collection 1: "geo_chunks" (Our Proposed Approach)
     * Indexes atomic Q&A units with BLUF headers, clinical inquiry, and citation metadata.
   - Collection 2: "fixed_chunks" (Comparative Baseline)
     * Indexes standard sliding-window fixed-size chunks (500 chars).
3. Metadata Persistence:
   - Persists document_id, source institute, URL, clinical focus, question type,
     and fragmentation flags to empower real-time verifiable citation proof cards.
================================================================================
"""

import os
import sys
import json
import time
import logging
from pathlib import Path
from typing import List, Dict, Any, Optional, Tuple
import chromadb
from chromadb.config import Settings
from sentence_transformers import SentenceTransformer

logging.basicConfig(
    level=logging.INFO,
    format="%(asctime)s [%(levelname)s] %(name)s: %(message)s",
    handlers=[logging.StreamHandler(sys.stdout)]
)
logger = logging.getLogger("Indexing")

DEFAULT_EMBEDDING_MODEL = "sentence-transformers/all-MiniLM-L6-v2"
GEO_COLLECTION_NAME = "geo_chunks"
FIXED_COLLECTION_NAME = "fixed_chunks"


class ClinicalVectorIndex:
    """
    Manages embedding generation and dual-collection persistence in ChromaDB.
    """

    def __init__(
        self,
        persist_dir: Optional[Path] = None,
        model_name: str = DEFAULT_EMBEDDING_MODEL
    ):
        """
        Initializes the ChromaDB persistent client and loads the embedding model.
        """
        if persist_dir is None:
            project_root = Path(__file__).resolve().parent.parent
            persist_dir = project_root / "chroma_db"

        self.persist_dir = Path(persist_dir)
        self.persist_dir.mkdir(parents=True, exist_ok=True)
        self.model_name = model_name

        logger.info(f"Connecting to persistent ChromaDB at: {self.persist_dir}")
        self.client = chromadb.PersistentClient(
            path=str(self.persist_dir),
            settings=Settings(anonymized_telemetry=False, allow_reset=True)
        )

        logger.info(f"Loading SentenceTransformer embedding model: {self.model_name}...")
        self.embedder = SentenceTransformer(self.model_name)
        logger.info("Embedding model loaded successfully.")

        # Initialize or retrieve both collections with cosine space
        self.geo_collection = self.client.get_or_create_collection(
            name=GEO_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "description": "GEO-structured Q&A with BLUF headers"}
        )
        self.fixed_collection = self.client.get_or_create_collection(
            name=FIXED_COLLECTION_NAME,
            metadata={"hnsw:space": "cosine", "description": "Baseline fixed-size 500-char chunks"}
        )

    def index_chunks(
        self,
        chunks: List[Dict[str, Any]],
        collection_name: str,
        batch_size: int = 100,
        reset_existing: bool = False
    ) -> int:
        """
        Encodes and indexes knowledge chunks into the specified ChromaDB collection.

        Args:
            chunks: List of chunk dictionaries.
            collection_name: Name of the collection ("geo_chunks" or "fixed_chunks").
            batch_size: Batch size for encoding and insertion (optimized for CPU/RAM).
            reset_existing: If True, clears existing items before indexing.

        Returns:
            Total number of items indexed.
        """
        target_collection = (
            self.geo_collection if collection_name == GEO_COLLECTION_NAME else self.fixed_collection
        )

        if reset_existing:
            logger.info(f"Resetting existing records in collection '{collection_name}'...")
            self.client.delete_collection(name=collection_name)
            target_collection = self.client.create_collection(
                name=collection_name,
                metadata={"hnsw:space": "cosine"}
            )
            if collection_name == GEO_COLLECTION_NAME:
                self.geo_collection = target_collection
            else:
                self.fixed_collection = target_collection

        existing_count = target_collection.count()
        if existing_count >= len(chunks) and not reset_existing:
            logger.info(
                f"Collection '{collection_name}' already populated with {existing_count} records. Skipping indexing."
            )
            return existing_count

        total_chunks = len(chunks)
        logger.info(f"Indexing {total_chunks} records into collection '{collection_name}'...")
        t0 = time.perf_counter()

        for i in range(0, total_chunks, batch_size):
            batch = chunks[i : i + batch_size]

            ids = [b["chunk_id"] for b in batch]
            documents = [b["content"] for b in batch]

            # Clean and sanitize metadata for ChromaDB (scalars only)
            metadatas = []
            for b in batch:
                meta = {
                    "document_id": str(b.get("document_id", "UNKNOWN")),
                    "source": str(b.get("source", "NIH")),
                    "url": str(b.get("url", "")),
                    "focus": str(b.get("focus", "")),
                    "strategy": str(b.get("strategy", collection_name)),
                    "is_fragmented": bool(b.get("is_fragmented", False)),
                    "char_length": int(b.get("char_length", len(b["content"])))
                }
                if "question_type" in b:
                    meta["question_type"] = str(b["question_type"])
                if "bluf" in b:
                    meta["bluf"] = str(b["bluf"])[:300]
                metadatas.append(meta)

            # Generate dense embeddings
            embeddings = self.embedder.encode(
                documents,
                batch_size=batch_size,
                show_progress_bar=False,
                convert_to_numpy=True,
                normalize_embeddings=True
            ).tolist()

            target_collection.upsert(
                ids=ids,
                documents=documents,
                embeddings=embeddings,
                metadatas=metadatas
            )
            logger.info(f"  Indexed batch {min(i + batch_size, total_chunks)}/{total_chunks}...")

        elapsed = time.perf_counter() - t0
        logger.info(
            f"Successfully indexed {total_chunks} chunks into '{collection_name}' in {elapsed:.2f} seconds."
        )
        return target_collection.count()

    def query(
        self,
        query_text: str,
        collection_name: str = GEO_COLLECTION_NAME,
        top_k: int = 3
    ) -> List[Dict[str, Any]]:
        """
        Performs dense semantic vector search against a specified collection.

        Args:
            query_text: Sanitized patient inquiry.
            collection_name: "geo_chunks" or "fixed_chunks".
            top_k: Number of nearest neighbors to retrieve.

        Returns:
            List of structured result dictionaries with cosine similarity scores.
        """
        target_collection = (
            self.geo_collection if collection_name == GEO_COLLECTION_NAME else self.fixed_collection
        )

        # Generate query vector
        query_embedding = self.embedder.encode(
            [query_text],
            convert_to_numpy=True,
            normalize_embeddings=True
        ).tolist()

        raw_results = target_collection.query(
            query_embeddings=query_embedding,
            n_results=top_k,
            include=["documents", "metadatas", "distances"]
        )

        formatted_results: List[Dict[str, Any]] = []
        if not raw_results or not raw_results["ids"]:
            return formatted_results

        ids = raw_results["ids"][0]
        documents = raw_results["documents"][0]
        metadatas = raw_results["metadatas"][0]
        distances = raw_results["distances"][0]

        for chunk_id, doc, meta, dist in zip(ids, documents, metadatas, distances):
            # In Chroma cosine distance: similarity = 1 - distance
            cosine_similarity = max(0.0, min(1.0, 1.0 - dist))
            formatted_results.append({
                "chunk_id": chunk_id,
                "document": doc,
                "metadata": meta,
                "cosine_distance": round(float(dist), 4),
                "cosine_similarity": round(float(cosine_similarity), 4),
                "strategy": collection_name
            })

        return formatted_results


def run_indexing_pipeline(sample_limit: Optional[int] = 500):
    """
    Driver function for Deliverable 5:
    1. Loads GEO chunks and baseline fixed-size chunks.
    2. Builds dual persistent collections in ChromaDB.
    3. Runs a comparative test retrieval using the Rahul motivating scenario.
    """
    project_root = Path(__file__).resolve().parent.parent
    processed_dir = project_root / "data" / "processed"
    geo_file = processed_dir / "geo_structured_chunks.json"
    fixed_file = processed_dir / "baseline_fixed_chunks.json"

    if not geo_file.exists() or not fixed_file.exists():
        logger.error("Chunk files not found! Please run geo_structuring.py first.")
        return

    logger.info("Loading chunk datasets for indexing...")
    with open(geo_file, "r", encoding="utf-8") as f:
        geo_chunks = json.load(f)

    with open(fixed_file, "r", encoding="utf-8") as f:
        fixed_chunks = json.load(f)

    # For fast student laptop indexing during demo, index top sample or full
    if sample_limit:
        geo_to_index = geo_chunks[:sample_limit]
        fixed_to_index = fixed_chunks[:sample_limit]
    else:
        geo_to_index = geo_chunks
        fixed_to_index = fixed_chunks

    index_mgr = ClinicalVectorIndex()

    # Index Collection 1: GEO Chunks
    logger.info("Indexing Collection 1: GEO Structured Chunks...")
    index_mgr.index_chunks(geo_to_index, collection_name=GEO_COLLECTION_NAME, batch_size=64)

    # Index Collection 2: Fixed-Size Baseline Chunks
    logger.info("Indexing Collection 2: Baseline Fixed-Size Chunks...")
    index_mgr.index_chunks(fixed_to_index, collection_name=FIXED_COLLECTION_NAME, batch_size=64)

    # Smoke Test: Comparative Retrieval on the Rahul Motivating Query
    test_query = "[REDACTED_USER] Blood sugar is 220 mg/dL guidance"

    print("\n" + "=" * 80)
    print("INDEXING VERIFICATION & RETRIEVAL SMOKE TEST (RAHUL MOTIVATING QUERY)")
    print("=" * 80)
    print(f"Sanitized Query: '{test_query}'\n")

    print("--- [COLLECTION 1: GEO STRUCTURED CHUNKS (OURS)] ---")
    geo_results = index_mgr.query(test_query, collection_name=GEO_COLLECTION_NAME, top_k=2)
    for idx, r in enumerate(geo_results, start=1):
        meta = r["metadata"]
        print(f"Result #{idx} | Similarity: {r['cosine_similarity']:.4f} | NIH Doc ID: {meta.get('document_id')} ({meta.get('source')})")
        print(f"  Focus Topic : {meta.get('focus')}")
        print(f"  BLUF Header : {meta.get('bluf', 'N/A')[:130]}...")
        print(f"  URL Link    : {meta.get('url')}")
        print(f"  Snippet     : {r['document'][:180]}...\n")

    print("--- [COLLECTION 2: BASELINE FIXED-SIZE CHUNKS] ---")
    fixed_results = index_mgr.query(test_query, collection_name=FIXED_COLLECTION_NAME, top_k=2)
    for idx, r in enumerate(fixed_results, start=1):
        meta = r["metadata"]
        print(f"Result #{idx} | Similarity: {r['cosine_similarity']:.4f} | Fragmented: {meta.get('is_fragmented')}")
        print(f"  Snippet     : {r['document'][:180]}...\n")
    print("=" * 80 + "\n")


if __name__ == "__main__":
    run_indexing_pipeline(sample_limit=500)
