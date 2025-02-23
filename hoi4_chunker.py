#!/usr/bin/env python3
"""
hoi4_chunker.py
================

A chunking library for HOI4 modding texts with enhanced HOI4 syntax awareness,
dual chunking output, and unified UUID generation with logging.

Deprecated:
  - The Rechunker and HOI4ChunkerDebug functions are no longer used in production.
  - UUID creation is now centralized here.
"""

import re
import os
import json
import uuid
import shutil
from datetime import datetime

# -----------------------------------------------------------------------------
# HOI4ChunkerConfig: Configuration for controlling chunking behavior.
# -----------------------------------------------------------------------------
class HOI4ChunkerConfig:
    def __init__(self, max_words=500, forbidden_start_tokens=None):
        if forbidden_start_tokens is None:
            forbidden_start_tokens = {
                "demonstrative": ["this", "that", "these", "those"],
                "conjunction": ["and", "but", "or", "nor", "for", "yet", "so"],
                "pronoun": ["it", "its", "they", "them", "their"]
            }
        self.max_words = max_words
        self.forbidden_start_tokens = forbidden_start_tokens

    def all_forbidden_tokens(self):
        tokens = []
        for key in self.forbidden_start_tokens:
            tokens.extend(self.forbidden_start_tokens[key])
        return set(token.lower() for token in tokens)

# -----------------------------------------------------------------------------
# HOI4Chunker: Splits text into dual sets of chunks (small and context).
# -----------------------------------------------------------------------------
class HOI4Chunker:
    def __init__(self, config=None):
        self.config = config if config is not None else HOI4ChunkerConfig()
        self.context_max_words = self.config.max_words

    def chunk_text_dual(self, text):
        """
        Process the input text and produce two lists of chunks:
          1. Standard chunks: smaller fragments.
          2. Context chunks: larger contiguous passages.
        """
        lines = text.splitlines()
        context_chunks = []
        small_chunks = []
        context_buffer = []
        mode = "narrative"
        brace_depth = 0
        sentence_split_re = re.compile(r'(?<=[.!?])\s+')

        def flush_context_buffer():
            nonlocal context_buffer
            if context_buffer:
                paragraph = "\n".join(context_buffer).strip()
                if paragraph:
                    words = paragraph.split()
                    if len(words) > self.context_max_words:
                        paragraph = " ".join(words[:self.context_max_words]) + " ..."
                    context_chunks.append(paragraph)
                context_buffer = []

        for line in lines:
            stripped = line.strip()
            if re.match(r'^\s*=+', stripped) and mode == "narrative":
                flush_context_buffer()
                context_chunks.append(stripped)
                small_chunks.append(stripped)
                continue
            if "{" in line:
                if mode == "narrative":
                    flush_context_buffer()
                    mode = "code"
                    brace_depth = 0
                brace_depth += line.count("{")
                brace_depth -= line.count("}")
                context_buffer.append(line)
                if stripped:
                    small_chunks.append(stripped)
                if brace_depth <= 0:
                    flush_context_buffer()
                    mode = "narrative"
                    brace_depth = 0
                continue
            if mode == "code":
                brace_depth += line.count("{")
                brace_depth -= line.count("}")
                context_buffer.append(line)
                if stripped:
                    small_chunks.append(stripped)
                if brace_depth <= 0:
                    flush_context_buffer()
                    mode = "narrative"
                    brace_depth = 0
                continue
            if not stripped:
                flush_context_buffer()
                continue
            context_buffer.append(line)
            sentences = sentence_split_re.split(stripped)
            for sentence in sentences:
                sentence = sentence.strip()
                if sentence:
                    small_chunks.append(sentence)
        flush_context_buffer()
        context_chunks = self.post_process_chunks(context_chunks)
        small_chunks = self.post_process_chunks(small_chunks)
        return small_chunks, context_chunks

    def post_process_chunks(self, chunks):
        forbidden = self.config.all_forbidden_tokens()
        processed = []
        for chunk in chunks:
            if chunk.startswith('=') or ('{' in chunk or '}' in chunk):
                processed.append(chunk)
                continue
            tokens = chunk.split()
            if tokens:
                first_token = tokens[0].strip(",.").lower()
                if processed and first_token in forbidden:
                    processed[-1] = processed[-1] + "\n" + chunk
                    continue
            processed.append(chunk)
        return processed

    def chunk_text_dual_with_uuid(self, text, source_identifier="XYZ.abc"):
        """
        Process the input text using dual-chunking, assign a deterministic UUID to each chunk,
        log the mapping (UUID, source, type, timestamp) to data_log.json, and return the chunks
        as lists of dictionaries.
        """
        small_chunks, context_chunks = self.chunk_text_dual(text)
        standard_with_uuid = []
        context_with_uuid = []
        current_timestamp = datetime.now().isoformat()
        for chunk in small_chunks:
            chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk))
            standard_with_uuid.append({"uuid": chunk_uuid, "content": chunk})
            self._log_chunk_uuid(chunk_uuid, source_identifier, "standard", current_timestamp)
        for chunk in context_chunks:
            chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk))
            context_with_uuid.append({"uuid": chunk_uuid, "content": chunk})
            self._log_chunk_uuid(chunk_uuid, source_identifier, "context", current_timestamp)
        return standard_with_uuid, context_with_uuid

    def _log_chunk_uuid(self, chunk_uuid, source_identifier, chunk_type, timestamp):
        """
        Log a single chunk's UUID mapping to data_log.json.
        """
        log_file = "data_log.json"
        if os.path.exists(log_file):
            with open(log_file, "r", encoding="utf-8") as f:
                try:
                    log_data = json.load(f)
                except Exception:
                    log_data = []
        else:
            log_data = []
        log_entry = {
            "uuid": chunk_uuid,
            "source": source_identifier,
            "chunk_type": chunk_type,
            "timestamp": timestamp
        }
        log_data.append(log_entry)
        with open(log_file, "w", encoding="utf-8") as f:
            json.dump(log_data, f, indent=2)

# Deprecated classes (Rechunker, HOI4ChunkerDebug) are omitted in production.
if __name__ == "__main__":
    print("Running HOI4 Chunker in debug mode.")
    # Debug usage is deprecated.

import logging
import os
import json
import uuid
import shutil
from datetime import datetime

# Configure logging to display info-level messages
logging.basicConfig(level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s")

# -------------------------------------------------------------------------
# Rechunker: Iteratively reconstructs databases of chunks (both standard and context),
# now with support for exporting a protected VIL database to text.
# -------------------------------------------------------------------------

# !!!!!!!!! WARNING: RECHUNKER IS DEBUG ONLY AND DEPRECATED AS OF UUID UPDATES !!!!!!!!!!! #

class Rechunker:
    def __init__(self, source_file, db_file="hoi4_chunks_db.json", checkpoint_file="processed_chunks.json",
                 context_db_file=None, context_checkpoint_file=None, chunker=None):
        """
        Initialize the Rechunker with source file and database/checkpoint paths.
        Two sets of files are maintained: one for standard chunks and one for context chunks.

        Args:
            source_file (str): Path to the raw HOI4 text file OR VIL database directory.
            db_file (str): JSON file for standard chunks (default: "hoi4_chunks_db.json").
            checkpoint_file (str): JSON file recording processed UUIDs for standard chunks.
            context_db_file (str): JSON file for context chunks (default: db_file with "_context" appended).
            context_checkpoint_file (str): JSON file for context chunk UUIDs.
            chunker (HOI4Chunker): Instance of HOI4Chunker. If None, a new instance is created.
        """
        self.source_file = source_file
        self.db_file = db_file
        self.checkpoint_file = checkpoint_file
        self.context_db_file = context_db_file if context_db_file is not None else db_file.replace(".json",
                                                                                                   "_context.json")
        self.context_checkpoint_file = context_checkpoint_file if context_checkpoint_file is not None else checkpoint_file.replace(
            ".json", "_context.json")
        self.chunker = chunker if chunker is not None else HOI4Chunker()

        # Save the original source_file (needed later if it's a protected directory)
        self.original_source = source_file

        # Load checkpoints and databases for standard chunks.
        if os.path.exists(self.checkpoint_file):
            with open(self.checkpoint_file, "r", encoding="utf-8") as f:
                self.processed_uuids = json.load(f)
        else:
            self.processed_uuids = []
        if os.path.exists(self.db_file):
            with open(self.db_file, "r", encoding="utf-8") as f:
                self.database = json.load(f)
        else:
            self.database = []

        # Load checkpoints and databases for context chunks.
        if os.path.exists(self.context_checkpoint_file):
            with open(self.context_checkpoint_file, "r", encoding="utf-8") as f:
                self.context_processed_uuids = json.load(f)
        else:
            self.context_processed_uuids = []
        if os.path.exists(self.context_db_file):
            with open(self.context_db_file, "r", encoding="utf-8") as f:
                self.context_database = json.load(f)
        else:
            self.context_database = []

    def backup_database(self, filename):
        """
        Backup a database file with a timestamp into a dedicated backup directory.
        """
        backup_dir = "db_backups"
        os.makedirs(backup_dir, exist_ok=True)
        if os.path.exists(filename):
            backup_name = os.path.join(backup_dir,
                                       f"{os.path.basename(filename)}.backup_{datetime.now().strftime('%Y%m%d_%H%M%S')}")
            shutil.copy(filename, backup_name)
            logging.info("Backup created: %s", backup_name)

    def export_vil_db_to_text(self):
        """
        Use VIL (HOI4VectorDB) to export the protected database into a continuous text file.
        Each exported entry includes its UUID.
        """
        try:
            from VIL import HOI4VectorDB  # Import the VIL module
        except ImportError:
            logging.error("Failed to import VIL. Make sure VIL.py is in your path.")
            return None

        logging.info("Exporting VIL database '%s' to text...", self.original_source)
        db_instance = HOI4VectorDB(db_name=self.original_source)
        # Query the collection without passing "ids" in the include list
        results = db_instance.collection.get(include=["documents", "embeddings"])
        docs = results.get("documents", [])
        ids = results.get("ids", [])
        lines = []
        for doc_id, doc in zip(ids, docs):
            lines.append(f"UUID: {doc_id}\n{doc}\n")
        export_filename = f"exported_{os.path.basename(self.original_source)}_{datetime.now().strftime('%Y%m%d_%H%M%S')}.txt"
        with open(export_filename, "w", encoding="utf-8") as f:
            f.write("\n".join(lines))
        logging.info("Export complete. Exported file: %s", export_filename)
        return export_filename

    def run_rechunking(self, iteration_limit=5):
        """
        Process the source file iteratively, adding up to iteration_limit new chunks for each database.
        New chunks are assigned deterministic UUIDs (using uuid5 based on content) with a rechunked_timestamp.
        If the source_file is a protected VIL database (directory), export it to text first.

        Args:
            iteration_limit (int): Maximum number of new chunks to process per iteration.
        """
        logging.info("Starting rechunking process with iteration limit %d", iteration_limit)
        # Backup both databases.
        self.backup_database(self.db_file)
        self.backup_database(self.context_db_file)

        # Determine if source_file is a file or a directory.
        if os.path.isdir(self.source_file):
            logging.info("Source '%s' is a directory (protected VIL database). Exporting to text...", self.source_file)
            exported_file = self.export_vil_db_to_text()
            if exported_file is None:
                logging.error("Export failed. Aborting rechunking process.")
                return
            self.source_file = exported_file

        # Try reading the source file.
        try:
            with open(self.source_file, "r", encoding="utf-8") as f:
                text = f.read()
        except Exception as e:
            logging.error("Failed to read source file '%s': %s", self.source_file, str(e))
            return

        standard_chunks, context_chunks = self.chunker.chunk_text_dual(text)
        new_standard = []
        count_std = 0
        current_timestamp = datetime.now().isoformat()
        for chunk in standard_chunks:
            chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk))
            if chunk_uuid in self.processed_uuids:
                continue
            self.processed_uuids.append(chunk_uuid)
            new_standard.append({"uuid": chunk_uuid, "content": chunk, "rechunked_timestamp": current_timestamp})
            count_std += 1
            if count_std >= iteration_limit:
                break

        new_context = []
        count_ctx = 0
        for chunk in context_chunks:
            chunk_uuid = str(uuid.uuid5(uuid.NAMESPACE_DNS, chunk))
            if chunk_uuid in self.context_processed_uuids:
                continue
            self.context_processed_uuids.append(chunk_uuid)
            new_context.append({"uuid": chunk_uuid, "content": chunk, "rechunked_timestamp": current_timestamp})
            count_ctx += 1
            if count_ctx >= iteration_limit:
                break

        self.database.extend(new_standard)
        self.context_database.extend(new_context)

        with open(self.db_file, "w", encoding="utf-8") as f:
            json.dump(self.database, f, indent=2)
        with open(self.checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(self.processed_uuids, f, indent=2)
        with open(self.context_db_file, "w", encoding="utf-8") as f:
            json.dump(self.context_database, f, indent=2)
        with open(self.context_checkpoint_file, "w", encoding="utf-8") as f:
            json.dump(self.context_processed_uuids, f, indent=2)

        logging.info("Processed %d new standard chunk(s). Total standard chunks: %d", count_std, len(self.database))
        logging.info("Processed %d new context chunk(s). Total context chunks: %d", count_ctx,
                     len(self.context_database))
        if count_std < iteration_limit and count_ctx < iteration_limit:
            logging.info("Rechunking complete; no more new chunks found.")

        # If the original source was a protected VIL database, delete it after export.
        if os.path.isdir(self.original_source):
            try:
                logging.info("Deleting old VIL database directory: %s", self.original_source)
                shutil.rmtree(self.original_source, ignore_errors=True)
                logging.info("Old VIL database deleted successfully.")
            except Exception as e:
                logging.error("Failed to delete old VIL database '%s': %s", self.original_source, str(e))

# -----------------------------------------------------------------------------
# HOI4ChunkerDebug: Debug menu for testing dual chunking and rechunking functionality.
# -----------------------------------------------------------------------------
class HOI4ChunkerDebug:
    def __init__(self):
        self.chunker = HOI4Chunker()

    def display_menu(self):
        """
        Display a command-line debug menu allowing the user to:
          1. View both standard and context chunked sections from a .txt file.
          2. Run the rechunking process.
          3. Exit the menu.
        """
        while True:
            print("\n=== HOI4 Chunker Debug Menu ===")
            print("1. Display chunked sections from a .txt file")
            print("2. Run rechunking process")
            print("3. Exit")
            choice = input("Enter your choice (1-3): ").strip()

            if choice == "1":
                file_path = input("Enter the path to the .txt file: ").strip()
                if os.path.exists(file_path):
                    with open(file_path, "r", encoding="utf-8") as f:
                        text = f.read()
                    standard, context = self.chunker.chunk_text_dual(text)
                    print("\n--- Standard (Small) Chunks ---")
                    for i, chunk in enumerate(standard):
                        print(f"\n--- Chunk {i+1} ---\n{chunk}")
                    print("\n--- Context Chunks ---")
                    for i, chunk in enumerate(context):
                        print(f"\n--- Chunk {i+1} ---\n{chunk}")
                else:
                    print("File does not exist.")
            elif choice == "2":
                source_file = input("Enter the path to the source .txt file for rechunking: ").strip()
                iter_input = input("Enter number of chunks to process per iteration (default 5): ").strip()
                try:
                    iteration = int(iter_input)
                except:
                    iteration = 5
                rechunker = Rechunker(source_file, chunker=self.chunker)
                rechunker.run_rechunking(iteration_limit=iteration)
            elif choice == "3":
                print("Exiting debug menu.")
                break
            else:
                print("Invalid choice. Please select 1, 2, or 3.")

# -----------------------------------------------------------------------------
# Debug execution: Only run the debug menu when the file is executed directly.
# -----------------------------------------------------------------------------
if __name__ == "__main__":
    print("Running HOI4 Chunker in debug mode.")
    debug_menu = HOI4ChunkerDebug()
    debug_menu.display_menu()
