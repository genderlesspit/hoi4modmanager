#!/usr/bin/env python3
"""
HOI4 Vector Ingestion Library (Debug Mode)
===========================================
This module is a library for managing a HOI4 modding vector database. It is
designed for debug purposes and includes three main classes:

1. HOI4VectorDB:
   - Creates and manages a persistent ChromaDB vector database.
   - Methods:
       __init__(db_name: str, embed_model: str):
           Initializes the database and loads the SentenceTransformer model.
       embed_text(text: str) -> list:
           Generates a vector embedding for a given text.

2. HOI4VectorUploader:
   - Processes and uploads raw text or JSON data to the vector database.
   - Methods:
       process_hoi4_data(raw_data) -> list:
           Processes raw HOI4 modding data into text entries.
       add_data(raw_data, metadata: dict = None):
           Processes and uploads data while automatically generating unique IDs.
       ingest_from_json(file_path: str):
           Loads and ingests data from a JSON file.
       ingest_from_text(text: str):
           Ingests raw text directly.

3. HOI4VectorDebug:
   - A debugging tool that retrieves and displays a selection of the database entries.
   - Methods:
       retrieve_first_five_entries() -> None:
           Retrieves and prints the first five stored entries (documents) along with their IDs.

Logging:
   - Timestamped logs are written to a dedicated "logs" directory.
   - Each execution generates a new log file with a unique timestamp.

Note:
   - This file is intended to be used as a library.
   - When executed directly (debug mode), the user is prompted for:
         • The debug database name.
         • Test data to ingest.
   - The debug functions explicitly create and modify the specified debug database.
"""

import os
import json
import uuid
import logging
from datetime import datetime
import chromadb
from sentence_transformers import SentenceTransformer

# ------------------------------------------------------------------------------
# Setup timestamped logging in a dedicated logs directory
# ------------------------------------------------------------------------------
log_dir = "logs"
if not os.path.exists(log_dir):
    os.makedirs(log_dir)
timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
log_file = os.path.join(log_dir, f"hoi4_vector_ingest_{timestamp}.log")
logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def console_log(message):
    """
    Logs and prints a message with a timestamp.

    :param message: The message string to log and print.
    """
    print(message)
    logging.info(message)


# ------------------------------------------------------------------------------
# HOI4VectorDB: Class for creating and managing the vectored database
# ------------------------------------------------------------------------------
class HOI4VectorDB:
    """
    Manages a persistent ChromaDB vectored database for HOI4 modding data.

    Attributes:
        client (chromadb.PersistentClient): The ChromaDB client instance.
        collection (chromadb.Collection): The collection within the database.
        embed_model (SentenceTransformer): The model for generating text embeddings.
    """

    def __init__(self, db_name="hoi4_vector_db", embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        """
        Initializes the vectored database and loads the embedding model.

        :param db_name: Name of the database.
        :param embed_model: Identifier for the SentenceTransformer model.
        """
        self.db_name = db_name
        self.client = chromadb.PersistentClient(path=f"./{db_name}")
        self.collection = self.client.get_or_create_collection(name=db_name)
        console_log(f"HOI4VectorDB initialized with database: {db_name}")
        console_log("Loading SentenceTransformer model...")
        self.embed_model = SentenceTransformer(embed_model)
        console_log(f"SentenceTransformer model '{embed_model}' loaded.")

    def embed_text(self, text):
        """
        Generates a vector embedding for the input text.

        :param text: Raw text to be embedded.
        :return: List of floats representing the text embedding.
        """
        console_log(f"Embedding text (first 50 chars): {text[:50]}...")
        embedding = self.embed_model.encode(text).tolist()
        console_log(f"Generated embedding (first 5 values): {embedding[:5]}")
        return embedding


# ------------------------------------------------------------------------------
# HOI4VectorUploader: Class for uploading raw text or JSON data
# ------------------------------------------------------------------------------
class HOI4VectorUploader:
    """
    Uploads raw HOI4 modding data to the vectored database.

    Methods:
        process_hoi4_data(raw_data) -> list:
            Processes raw HOI4 modding data into text entries.
        add_data(raw_data, metadata: dict = None):
            Processes and uploads data while automatically generating unique IDs.
        ingest_from_json(file_path: str):
            Loads and ingests data from a JSON file.
        ingest_from_text(text: str):
            Ingests raw text directly.
    """

    def __init__(self, vector_db: HOI4VectorDB):
        """
        Initializes the uploader with a HOI4VectorDB instance.

        :param vector_db: An instance of HOI4VectorDB.
        """
        self.vector_db = vector_db

    def process_hoi4_data(self, raw_data):
        """
        Processes raw HOI4 modding data into a list of text entries.

        :param raw_data: Data as a list of dicts, list of strings, or a single string.
        :return: List of processed text entries.
        """
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
        """
        Processes and uploads data to the vectored database while automatically generating unique IDs.

        :param raw_data: Raw data (list or string) to be ingested.
        :param metadata: Optional metadata for each entry.
        """
        processed_texts = self.process_hoi4_data(raw_data)
        if not processed_texts:
            console_log("❌ No valid data to ingest.")
            return
        embeddings = [self.vector_db.embed_text(text) for text in processed_texts]
        console_log("Adding data to the vectored database...")
        # Generate unique IDs for each entry using uuid
        ids = [str(uuid.uuid4()) for _ in range(len(processed_texts))]
        self.vector_db.collection.add(embeddings=embeddings, documents=processed_texts, metadatas=metadata, ids=ids)
        console_log(f"✅ {len(processed_texts)} entries added to ChromaDB")
        count = self.vector_db.collection.count()
        console_log(f"Database now contains {count} entries.")

    def ingest_from_json(self, file_path):
        """
        Loads JSON data from a file and uploads it to the vectored database.

        :param file_path: Path to the JSON file.
        """
        if not os.path.exists(file_path):
            console_log(f"❌ File not found: {file_path}")
            return
        console_log(f"Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)
        self.add_data(raw_data)
        console_log(f"🎉 Data from {file_path} ingested successfully.")

    def ingest_from_text(self, text):
        """
        Uploads raw text directly to the vectored database.

        :param text: Raw text to be ingested.
        """
        if not text.strip():
            console_log("❌ No text provided for ingestion.")
            return
        console_log("Ingesting raw text...")
        self.add_data(text)
        console_log("✅ Raw text successfully ingested into ChromaDB.")


# ------------------------------------------------------------------------------
# HOI4VectorDebug: Class for debugging and viewing stored database entries
# ------------------------------------------------------------------------------
class HOI4VectorDebug:
    """
    Debug tool for viewing stored data in the vectored database.

    Methods:
        retrieve_first_five_entries() -> None:
            Retrieves and prints the first five stored entries (documents) along with their IDs.
    """

    def __init__(self, vector_db: HOI4VectorDB):
        """
        Initializes the debug tool with a HOI4VectorDB instance.

        :param vector_db: An instance of HOI4VectorDB.
        """
        self.vector_db = vector_db

    def retrieve_first_five_entries(self):
        """
        Retrieves and prints the first five stored entries (documents) from the database.
        """
        count = self.vector_db.collection.count()
        console_log(f"Retrieving up to the first 5 of {count} stored entries from ChromaDB...")
        if count == 0:
            console_log("⚠️ No data found in the database.")
            return
        # Request only allowed include items. IDs are returned by default.
        results = self.vector_db.collection.get(include=['documents', 'embeddings'])
        docs = results.get("documents", [])[:5]
        ids = results.get("ids", [])[:5]  # IDs are returned automatically.
        for idx, (doc_id, doc) in enumerate(zip(ids, docs)):
            console_log(f"Entry {idx + 1} (ID: {doc_id}): {doc}")


# ------------------------------------------------------------------------------
# Debug execution: This block runs only when the file is executed directly.
# ------------------------------------------------------------------------------
if __name__ == "__main__":
    console_log("=== HOI4 Vector Ingestion Library Debug Mode ===")

    # Prompt user for a dynamic debug database name
    debug_db_name = input(
        "Enter debug database name (default 'debug_hoi4_vector_db'): ").strip() or "debug_hoi4_vector_db"
    console_log(f"Using debug database: {debug_db_name}")

    # Create a new HOI4VectorDB with the specified debug database name
    vector_db = HOI4VectorDB(db_name=debug_db_name)

    # Create an uploader using the debug database instance
    uploader = HOI4VectorUploader(vector_db)

    # Prompt the user for test data input (raw text)
    test_data = input("Enter test data (raw text) to ingest: ").strip()
    if test_data:
        uploader.ingest_from_text(test_data)
    else:
        console_log("No test data provided; skipping ingestion.")

    # Create a debug instance to view the database
    debugger = HOI4VectorDebug(vector_db)
    debugger.retrieve_first_five_entries()

    console_log("=== HOI4 Vector Ingestion Library Debug Completed ===")
