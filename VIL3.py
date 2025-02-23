#!/usr/bin/env python3
"""
HOI4 Vector Ingestion Library (Unified UUID Mode)
===========================================
This module manages a HOI4 modding vector database and now depends on the chunker
for deterministic UUID generation and logging.

Main changes:
  • HOI4VectorUploader.ingest_from_text_dual now uses chunk_text_dual_with_uuid.
  • New method ingest_chunk embeds a chunk and adds it to the database using the chunker’s UUID.
"""

import os
import json
import uuid
import logging
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

# NEW: Import the dual-chunking classes from hoi4_chunker.py
from hoi4_chunker import HOI4Chunker, HOI4ChunkerConfig

import logging
from datetime import datetime

log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
log_filename = os.path.join(log_dir, f"hoi4_vector_ingest_{datetime.now().strftime('%Y%m%d_%H%M%S')}.log")
logging.basicConfig(
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s",
    handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
)

def console_log(message):
    print(f"{datetime.now().strftime('%Y-%m-%d %H:%M:%S')} - {message}")
    logging.info(message)

# ------------------------------------------------------------------------------
# HOI4VectorDB: Manages a persistent ChromaDB vector database.
# ------------------------------------------------------------------------------
class HOI4VectorDB:
    def __init__(self, db_name="hoi4_vector_db", embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        self.db_name = db_name
        self.client = chromadb.PersistentClient(path=f"./{db_name}")
        self.collection = self.client.get_or_create_collection(name=db_name)
        console_log(f"HOI4VectorDB initialized with database: {db_name}")
        console_log("Loading SentenceTransformer model...")
        self.embed_model = SentenceTransformer(embed_model)
        console_log(f"SentenceTransformer model '{embed_model}' loaded.")

    def embed_text(self, text):
        console_log(f"Embedding text (first 50 chars): {text[:50]}...")
        embedding = self.embed_model.encode(text).tolist()
        console_log(f"Generated embedding (first 5 values): {embedding[:5]}")
        return embedding

# -------------------------------------------------------------------------------
# HOI4VectorUploader: Uploads raw HOI4 modding data using the unified UUID system.
# ------------------------------------------------------------------------------
class HOI4VectorUploader:
    def __init__(self, vector_db: HOI4VectorDB):
        self.vector_db = vector_db

    def process_hoi4_data(self, raw_data):
        console_log("Processing raw HOI4 modding data...")
        processed_data = []
        if isinstance(raw_data, list):
            for entry in raw_data:
                if isinstance(entry, dict) and "description" in entry:
                    processed_data.append(entry["description"])
                elif isinstance(entry, str):
                    processed_data.append(entry)
        elif isinstance(raw_data, str):
            processed_data.append(raw_data)
        console_log(f"Processed {len(processed_data)} entries.")
        return processed_data

    def add_data(self, raw_data, metadata=None):
        processed_texts = self.process_hoi4_data(raw_data)
        if not processed_texts:
            console_log("❌ No valid data to ingest.")
            return
        embeddings = [self.vector_db.embed_text(text) for text in processed_texts]
        console_log("Adding data to the vectored database...")
        ids = [str(uuid.uuid4()) for _ in range(len(processed_texts))]
        self.vector_db.collection.add(embeddings=embeddings, documents=processed_texts, metadatas=metadata, ids=ids)
        console_log(f"✅ {len(processed_texts)} entries added to ChromaDB")
        count = self.vector_db.collection.count()
        console_log(f"Database now contains {count} entries.")

    def ingest_chunk(self, chunk, uid, chunk_type):
        """
        Embed and add a single chunk to the vector database using the provided UUID.
        """
        embedding = self.vector_db.embed_text(chunk)
        self.vector_db.collection.add(
            embeddings=[embedding],
            documents=[chunk],
            metadatas={"chunk_type": chunk_type},
            ids=[uid]
        )
        console_log(f"Ingested {chunk_type} chunk with UUID: {uid}")

    def ingest_from_json(self, file_path):
        if not os.path.exists(file_path):
            console_log(f"❌ File not found: {file_path}")
            return
        console_log(f"Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.add_data(raw_data)
        console_log(f"🎉 Data from {file_path} ingested successfully.")

    def ingest_from_text(self, text):
        if not text.strip():
            console_log("❌ No text provided for ingestion.")
            return
        console_log("Ingesting raw text...")
        self.add_data(text)
        console_log("✅ Raw text successfully ingested into ChromaDB.")

    def ingest_from_text_dual(self, text):
        """
        Uses dual-chunking (with UUIDs from the chunker) to ingest both standard and context chunks.
        Returns a tuple of lists: (standard_chunk_UUIDs, context_chunk_UUIDs).
        """
        if not text.strip():
            console_log("❌ No text provided for dual-chunk ingestion.")
            return ([], [])
        console_log("Ingesting raw text with dual-chunking...")
        try:
            chunker = HOI4Chunker()  # Using default HOI4ChunkerConfig
            std_chunks, ctx_chunks = chunker.chunk_text_dual_with_uuid(text)
        except Exception as e:
            console_log(f"❌ Failed to initialize HOI4Chunker: {str(e)}")
            return ([], [])
        std_uuids = []
        for item in std_chunks:
            uid = item["uuid"]
            self.ingest_chunk(item["content"], uid, "standard")
            std_uuids.append(uid)
        ctx_uuids = []
        for item in ctx_chunks:
            uid = item["uuid"]
            self.ingest_chunk(item["content"], uid, "context")
            ctx_uuids.append(uid)
        console_log("✅ Dual-chunk text ingestion complete.")
        return (std_uuids, ctx_uuids)

# -------------------------------------------------------------------------------
# HOI4VectorDebug: Debug tool for viewing stored data.
# ------------------------------------------------------------------------------
class HOI4VectorDebug:
    def __init__(self, vector_db: HOI4VectorDB):
        self.vector_db = vector_db

    def retrieve_first_five_entries(self):
        count = self.vector_db.collection.count()
        console_log(f"Retrieving up to the first 5 of {count} stored entries from ChromaDB...")
        if count == 0:
            console_log("⚠️ No data found in the database.")
            return
        results = self.vector_db.collection.get(include=['documents', 'embeddings'])
        docs = results.get("documents", [])[:5]
        ids = results.get("ids", [])[:5]
        for idx, (doc_id, doc) in enumerate(zip(ids, docs)):
            console_log(f"Entry {idx + 1} (ID: {doc_id}): {doc}")

if __name__ == "__main__":
    console_log("=== HOI4 Vector Ingestion Library Dual-Chunk Debug Mode ===")
    debug_db_name = input("Enter debug database name (default 'debug_hoi4_vector_db'): ").strip() or "debug_hoi4_vector_db"
    import shutil
    if os.path.exists(debug_db_name):
        shutil.rmtree(debug_db_name)
        console_log(f"Existing database '{debug_db_name}' has been nuked.")
    else:
        console_log(f"No existing database '{debug_db_name}' found. Starting fresh.")
    vector_db = HOI4VectorDB(db_name=debug_db_name)
    uploader = HOI4VectorUploader(vector_db)
    test_data = input("Enter test data (raw text) to dual-chunk ingest: ").strip()
    if test_data:
        uploader.ingest_from_text_dual(test_data)
    else:
        console_log("No test data provided; skipping ingestion.")
    debugger = HOI4VectorDebug(vector_db)
    debugger.retrieve_first_five_entries()
    console_log("=== HOI4 Vector Ingestion Library Dual-Chunk Debug Completed ===")
