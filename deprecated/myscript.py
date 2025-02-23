import chromadb
import logging
import json
import os
import shutil
import numpy as np
from sentence_transformers import SentenceTransformer

# ✅ Set up logging
log_file = "hoi4_vector_ingest.log"
if os.path.exists(log_file):
    os.remove(log_file)  # Ensure log file resets on each run

logging.basicConfig(
    filename=log_file,
    level=logging.DEBUG,
    format="%(asctime)s - %(levelname)s - %(message)s"
)


def console_log(message):
    """Prints and logs messages for real-time progress visibility."""
    print(message)
    logging.info(message)


class HOI4VectorIngest:
    """Handles ingestion and transformation of HOI4 modding data into ChromaDB."""

    def __init__(self, db_name="hoi4_vector_db", embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the vector database and embedding model."""
        self.db_name = db_name
        self.client = chromadb.PersistentClient(path=f"./{db_name}")
        self.collection = self.client.get_or_create_collection(name=db_name)
        self.embed_model = SentenceTransformer(embed_model)
        console_log("HOI4VectorIngest initialized.")

    def embed_text(self, text):
        """Generate a vector embedding for the input text."""
        console_log(f"Embedding text: {text[:50]}...")
        embedding = self.embed_model.encode(text).tolist()
        console_log(f"🔢 Generated embedding (first 5 values): {embedding[:5]}")
        return embedding

    def process_hoi4_data(self, raw_data):
        """Transforms raw HOI4 modding data into structured text for embeddings."""
        console_log("Processing raw HOI4 modding data...")
        processed_data = []
        for entry in raw_data:
            if isinstance(entry, dict) and "description" in entry:
                processed_data.append(entry["description"])
            elif isinstance(entry, str):
                processed_data.append(entry)
        console_log(f"Processed {len(processed_data)} entries.")
        return processed_data

    def add_data(self, raw_data, metadata=None):
        """Add HOI4 modding knowledge to ChromaDB after processing."""
        console_log("Starting data ingestion...")
        processed_texts = self.process_hoi4_data(raw_data)
        embeddings = [self.embed_text(text) for text in processed_texts]
        ids = [f"id_{i}" for i in range(len(processed_texts))]

        self.collection.add(embeddings=embeddings, documents=processed_texts, metadatas=metadata, ids=ids)
        console_log(f"✅ {len(processed_texts)} entries added to ChromaDB")

    def ingest_from_json(self, file_path):
        """Load and process HOI4 modding data from a JSON file."""
        if not os.path.exists(file_path):
            console_log(f"❌ File not found: {file_path}")
            return

        console_log(f"📂 Loading data from {file_path}...")
        with open(file_path, "r", encoding="utf-8") as f:
            raw_data = json.load(f)

        self.add_data(raw_data)
        console_log(f"🎉 Data from {file_path} ingested successfully.")

    def ingest_from_text(self, text):
        """Ingest raw text directly into the vector database."""
        if not text.strip():
            console_log("❌ No text provided for ingestion.")
            return

        console_log("📥 Ingesting raw text...")
        self.add_data([text])
        console_log("✅ Raw text successfully ingested into ChromaDB.")

    def nuke_database(self):
        """Completely deletes the vector database and resets it."""
        console_log("🚨 Nuking database...")
        if os.path.exists(self.db_name):
            shutil.rmtree(self.db_name)
        self.client = chromadb.PersistentClient(path=f"./{self.db_name}")
        self.collection = self.client.get_or_create_collection(name=self.db_name)
        console_log("💥 Database has been completely reset.")


class HOI4VectorDebug:
    """Debugging class to retrieve stored data from ChromaDB."""

    def __init__(self, db_name="hoi4_vector_db"):
        """Initialize debug tool and connect to ChromaDB."""
        self.client = chromadb.PersistentClient(path=f"./{db_name}")
        self.collection = self.client.get_or_create_collection(name=db_name)
        console_log("HOI4VectorDebug initialized.")

    def retrieve_all_data(self):
        """Retrieve and display all stored documents in the database."""
        count = self.collection.count()
        console_log(f"🔍 Retrieving all {count} stored entries from ChromaDB...")

        if count == 0:
            console_log("⚠️ No data found in the database.")
            return []

        results = self.collection.get()
        console_log(f"Retrieved ChromaDB result: {results}")
        stored_texts = results.get("documents", [])
        stored_ids = results.get("ids", [])

        for idx, (doc_id, text) in enumerate(zip(stored_ids, stored_texts)):
            console_log(f"📜 Entry {idx + 1} (ID: {doc_id}): {text}")

        self.prompt_for_entry_details()

        return stored_texts

    def prompt_for_entry_details(self):
        """Prompt the user to enter an ID and display its vectored context."""
        entry_id = input("Enter an entry ID to view its vectored representation (or press Enter to skip): ").strip()
        if entry_id:
            self.retrieve_entry_by_id(entry_id)

    def retrieve_entry_by_id(self, entry_id):
        """Retrieve and display the raw vector embedding for a given entry ID."""
        result = self.collection.get(ids=[entry_id])
        console_log(f"🔍 Debugging retrieval for ID: {entry_id} | Full result: {result}")

        if not result.get("documents"):
            console_log(f"❌ No entry found with ID: {entry_id}")
            return

        embeddings = result.get("embeddings")
        if not embeddings or not embeddings[0]:
            console_log(f"⚠️ No embedding found for ID: {entry_id}. Data may not have been embedded correctly.")
            return

        console_log(
            f"📄 Entry ID: {entry_id}\n🔹 Document: {result['documents'][0]}\n🔢 Vector Embedding: {embeddings[0]}")


class HOI4VectorTest:
    """Test class for validating vector ingestion and retrieval."""

    def __init__(self):
        """Initialize test setup."""
        self.ingestor = HOI4VectorIngest()
        self.debugger = HOI4VectorDebug()

    def run_test(self):
        """Run test to validate ingestion and retrieval of test text."""
        test_text = """A focus tree is defined by using a focus_tree = { ... } block. The following arguments are used:

        id = my_focus_tree decides the ID that the focus tree uses. It is mandatory to define, and an overlap will result in an error. The ID is primarily used for the has_focus_tree trigger and the load_focus_tree effect, whose tooltips use it as a localization key.

        country = { ... } is a MTTH block that assigns a score for the focus tree, deciding which one is used in-game. This is evaluated before the game's start and the check is essentially never refreshed. The focus tree with the highest score will be the one that gets loaded for the country."""

        console_log("🚀 Running HOI4 Vector Ingest Test...")
        self.ingestor.ingest_from_text(test_text)
        console_log("✅ Test text ingested successfully.")

        console_log("🔍 Retrieving stored test entry...")
        self.debugger.retrieve_all_data()
        console_log("✅ Test retrieval completed.")


# ✅ Run test
if __name__ == "__main__":
    test_runner = HOI4VectorTest()
    test_runner.run_test()