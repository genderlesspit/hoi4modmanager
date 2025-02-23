#!/usr/bin/env python3
"""
HVLdb5 Debug Module (Synthesis with Refined Similarity and Dtype Fix)
======================================================================
This script synthesizes all updates from previous versions while ensuring that:
  • Embeddings used for similarity ranking are cast to float32 to prevent dtype mismatches.
  • Debug mode is prompted and timestamped logging is enabled.
  • Dual-chunk ingestion uses deterministic UUIDs (via HOI4Chunker) and vectorization via HOI4VectorDB.
  • Data directory integration ignores the integration log file, ingests new/updated files, deletes deprecated UUIDs,
    and logs the current count of entries in the vector database.
  • Query processing supports an adjustable similarity threshold and prints the final AI prompt in debug mode.

Note on Warnings:
  The “Add of existing embedding ID” warnings originate from the VIL ingestion process (via chromadb)
  when the same chunk ID is being added more than once. These warnings are benign and indicate that duplicate chunks
  are already present in the database. Adjust your ingestion logic if duplicates are undesired.
"""

import os
import time
import datetime
import logging
import json
import numpy as np
import torch
import transformers
import ollama
from transformers import AutoTokenizer

# Import our vector DB and uploader from VIL3, and our chunker from hoi4_chunker.
from VIL3 import HOI4VectorDB, HOI4VectorUploader
from hoi4_chunker import HOI4Chunker


# ---------------------------
# Framework Component Classes
# ---------------------------

class ChunkRanker:
    """
    Ranks standard (small) chunks by computing semantic similarity.
    Uses SentenceTransformer.similarity from the underlying model.
    """

    def __init__(self, vector_db):
        self.vector_db = vector_db

    def rank_chunks(self, query, top_k=5, similarity_threshold=0.75):
        logging.debug(f"Using similarity threshold: {similarity_threshold}")
        # Get query embedding from the SentenceTransformer model.
        query_embedding = self.vector_db.embed_model.encode(query)
        # Ensure the query embedding is float32.
        query_embedding = query_embedding.astype(np.float32)
        results = self.vector_db.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=top_k,
            where={"chunk_type": "standard"},
            include=["embeddings", "documents"]
        )
        if not results.get("documents") or not results["documents"][0]:
            logging.warning("No standard chunks found for the query.")
            return []
        docs = results["documents"][0]
        embeddings = np.array(results["embeddings"][0])
        # Convert embeddings to float32 to match query_embedding dtype.
        embeddings = embeddings.astype(np.float32)
        # Use the model's built-in similarity method.
        similarities = self.vector_db.embed_model.similarity(embeddings, [query_embedding])
        similarities = similarities.cpu().numpy().flatten()
        # Filter and sort chunks by similarity.
        ranked = [(None, doc, sim) for doc, sim in zip(docs, similarities) if sim >= similarity_threshold]
        ranked.sort(key=lambda x: x[2], reverse=True)
        logging.debug(f"Ranked {len(ranked)} standard chunks for the query.")
        return ranked


class ContextChunkRetriever:
    def __init__(self, vector_db):
        self.vector_db = vector_db

    def get_context_chunk(self, standard_chunk_text):
        query_embedding = self.vector_db.embed_model.encode(standard_chunk_text)
        query_embedding = query_embedding.astype(np.float32)
        results = self.vector_db.collection.query(
            query_embeddings=[query_embedding.tolist()],
            n_results=1,
            where={"chunk_type": "context"},
            include=["documents"]
        )
        if not results.get("documents") or not results["documents"][0]:
            logging.warning("No context chunk found for the provided standard chunk.")
            return ""
        return results["documents"][0][0]


class TokenLimitManager:
    def __init__(self, max_tokens=1500):
        self.max_tokens = max_tokens

    def count_tokens(self, text):
        return len(text.split())

    def accumulate_context(self, context_chunks):
        accumulated = ""
        for chunk in context_chunks:
            trial = accumulated + "\n" + chunk if accumulated else chunk
            if self.count_tokens(trial) <= self.max_tokens:
                accumulated = trial
            else:
                break
        return accumulated.strip()


class AIModule:
    def __init__(self, use_falcon=True):
        self.use_falcon = use_falcon
        if use_falcon:
            self.model_id = "tiiuae/falcon-7b-instruct"
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.pipeline = transformers.pipeline(
                "text-generation",
                model=self.model_id,
                tokenizer=self.tokenizer,
                torch_dtype=torch.bfloat16,
                trust_remote_code=True,
                device_map="auto"
            )
        else:
            self.model_id = "llama3"

    def generate_answer(self, prompt):
        start_time = time.time()
        if self.use_falcon:
            sequences = self.pipeline(prompt, max_length=200, do_sample=True, top_k=10, num_return_sequences=1)
            response = sequences[0]['generated_text']
        else:
            response = ollama.chat(
                model=self.model_id,
                messages=[{"role": "user", "content": prompt}]
            )["message"]["content"].strip()
        end_time = time.time()
        logging.debug(f"Generated answer in {end_time - start_time:.2f} sec | Prompt length: {len(prompt)} chars")
        return response


class QueryProcessor:
    def __init__(self, vector_db, ai_module, token_manager, chunk_ranker, context_retriever, debug=False):
        self.vector_db = vector_db
        self.ai_module = ai_module
        self.token_manager = token_manager
        self.chunk_ranker = chunk_ranker
        self.context_retriever = context_retriever
        self.debug = debug

    def process_query(self, user_query, similarity_threshold=0.75):
        logging.info(f"Processing query with similarity threshold: {similarity_threshold}")
        ranked_chunks = self.chunk_ranker.rank_chunks(user_query, similarity_threshold=similarity_threshold)
        if not ranked_chunks:
            return "⚠️ Insufficient data in the database to answer accurately."
        context_chunks = []
        for _, std_text, sim in ranked_chunks:
            context = self.context_retriever.get_context_chunk(std_text)
            if context:
                context_chunks.append(context)
        if not context_chunks:
            return "⚠️ Insufficient data in the database to answer accurately."
        accumulated_context = self.token_manager.accumulate_context(context_chunks)
        prompt = f"Relevant Context:\n{accumulated_context}\n\nUser Query:\n{user_query}"
        logging.debug(f"Final prompt for AI:\n{prompt}")
        if self.debug:
            print("\nDEBUG: Final prompt fed into AI:")
            print(prompt)
        attempts = 0
        while attempts < len(context_chunks):
            try:
                answer = self.ai_module.generate_answer(prompt)
                return answer
            except Exception as e:
                logging.error(f"AI generation error: {str(e)}. Reducing context and retrying.")
                if context_chunks:
                    context_chunks.pop()
                    accumulated_context = self.token_manager.accumulate_context(context_chunks)
                    prompt = f"Relevant Context:\n{accumulated_context}\n\nUser Query:\n{user_query}"
                    if self.debug:
                        print("\nDEBUG: Updated prompt fed into AI:")
                        print(prompt)
                    attempts += 1
                else:
                    return "Error: Unable to generate an answer due to token limits."
        return "Error: Unable to generate an answer after reducing context."


class DataDirectoryIntegrator:
    """
    Monitors a designated data directory for new, modified, or removed .txt/.json files.
    Uses HOI4VectorUploader (from VIL3) to ingest data via dual-chunking.
    Skips the integration log file itself.
    Logs the current count of entries in the vector database after processing.
    """

    def __init__(self, vector_db, data_dir="data", log_filename="data_integration_log.json"):
        self.vector_db = vector_db
        self.data_dir = data_dir
        if not os.path.exists(self.data_dir):
            os.makedirs(self.data_dir)
        self.log_file = os.path.join(self.data_dir, log_filename)
        if os.path.exists(self.log_file):
            with open(self.log_file, "r", encoding="utf-8") as f:
                self.integration_log = json.load(f)
        else:
            self.integration_log = {}

    def check_and_update(self):
        files = [f for f in os.listdir(self.data_dir)
                 if (f.endswith(".txt") or f.endswith(".json")) and f != "data_integration_log.json"]
        new_data_found = False
        uploader = HOI4VectorUploader(self.vector_db)
        for file in files:
            filepath = os.path.join(self.data_dir, file)
            modified_time = os.path.getmtime(filepath)
            record = self.integration_log.get(file)
            if record is None or record["modified_time"] < modified_time:
                logging.info(f"New or updated file detected: {file}")
                try:
                    if file.endswith(".txt"):
                        with open(filepath, "r", encoding="utf-8") as f:
                            raw_text = f.read()
                        std_uuids, ctx_uuids = uploader.ingest_from_text_dual(raw_text)
                        chunk_uuids = std_uuids + ctx_uuids
                    elif file.endswith(".json"):
                        uploader.ingest_from_json(filepath)
                        chunk_uuids = []  # Modify if JSON ingestion returns UUIDs.
                    if record is not None:
                        old_chunk_uuids = record.get("chunk_uuids", [])
                        deprecated = list(set(old_chunk_uuids) - set(chunk_uuids))
                        if deprecated:
                            logging.info(f"Removing deprecated UUIDs for updated file {file}: {deprecated}")
                            self.vector_db.collection.delete(ids=deprecated)
                    self.integration_log[file] = {
                        "modified_time": modified_time,
                        "uploaded_at": datetime.datetime.now().isoformat(),
                        "chunk_uuids": chunk_uuids
                    }
                    new_data_found = True
                    logging.info(
                        f"Integrated file {file} at {self.integration_log[file]['uploaded_at']} with chunk UUIDs: {chunk_uuids}")
                except Exception as e:
                    logging.error(f"Error integrating file {file}: {str(e)}")
        removed_files = [file for file in self.integration_log if file not in files]
        for file in removed_files:
            logging.info(f"File removed from data directory: {file}")
            uuids = self.integration_log[file]["chunk_uuids"]
            try:
                self.vector_db.collection.delete(ids=uuids)
                logging.info(f"Deleted chunks for removed file {file}: {uuids}")
            except Exception as e:
                logging.error(f"Error deleting chunks for removed file {file}: {str(e)}")
            del self.integration_log[file]
            new_data_found = True

        with open(self.log_file, "w", encoding="utf-8") as f:
            json.dump(self.integration_log, f, indent=2)
        db_count = self.vector_db.collection.count()
        logging.info(f"Vector database now contains {db_count} entries.")
        if new_data_found:
            logging.info("Data directory integration complete. New or updated data was processed.")
            return True
        else:
            logging.info("No new data found in the data directory.")
            return False


# ---------------------------
# Debug Interface & Execution Module
# ---------------------------
def setup_logging(debug_mode):
    logs_dir = "logs"
    if not os.path.exists(logs_dir):
        os.makedirs(logs_dir)
    timestamp = datetime.datetime.now().strftime("%Y%m%d_%H%M%S")
    log_filename = os.path.join(logs_dir, f"HVLdb5_debug_{timestamp}.log")
    log_level = logging.DEBUG if debug_mode else logging.INFO
    logging.basicConfig(
        level=log_level,
        format="%(asctime)s - %(levelname)s - %(message)s",
        handlers=[logging.FileHandler(log_filename), logging.StreamHandler()]
    )
    logging.info("Logging setup complete.")
    logging.info(f"Debug mode is {'ON' if debug_mode else 'OFF'}.")


class HVLdb5DebugInterface:
    def __init__(self, debug_mode=False):
        self.debug_mode = debug_mode
        setup_logging(self.debug_mode)
        logging.info("Initializing HVLdb5 Debug Interface.")
        self.vector_db = HOI4VectorDB(db_name="hoi4_vector_db")
        self.chunk_ranker = ChunkRanker(self.vector_db)
        self.context_retriever = ContextChunkRetriever(self.vector_db)
        self.token_manager = TokenLimitManager(max_tokens=1500)
        self.ai_module = AIModule(use_falcon=False)  # Change to True to use Falcon-7B.
        self.query_processor = QueryProcessor(
            self.vector_db,
            self.ai_module,
            self.token_manager,
            self.chunk_ranker,
            self.context_retriever,
            debug=self.debug_mode
        )
        self.data_integrator = DataDirectoryIntegrator(self.vector_db, data_dir="data",
                                                       log_filename="data_integration_log.json")
        logging.info("HVLdb5 Debug Interface initialized successfully.")

    def run(self):
        logging.info("Starting interactive debug session.")
        print("=== HVLdb5 Debug Interface ===")
        while True:
            user_query = input("\nEnter your HOI4 modding query (or type 'exit' to quit): ").strip()
            logging.info(f"User query: {user_query}")
            if user_query.lower() == "exit":
                logging.info("Exiting interactive debug session.")
                break

            logging.info("Performing data directory check before processing query...")
            self.data_integrator.check_and_update()

            similarity_threshold = 0.75
            answer = self.query_processor.process_query(user_query, similarity_threshold=similarity_threshold)
            if "Insufficient data" in answer:
                print("\nInsufficient data found. Would you like to lower the similarity threshold? (current: 0.75)")
                new_thresh = input(
                    "Enter new similarity threshold (e.g., 0.65) or press Enter to keep current: ").strip()
                if new_thresh:
                    try:
                        similarity_threshold = float(new_thresh)
                        logging.info(f"User adjusted similarity threshold to: {similarity_threshold}")
                        answer = self.query_processor.process_query(user_query,
                                                                    similarity_threshold=similarity_threshold)
                    except Exception as e:
                        logging.error(
                            f"Invalid threshold input: {new_thresh} ({str(e)}). Continuing with default threshold.")

            start_time = time.time()
            end_time = time.time()
            duration = end_time - start_time
            logging.info(f"Query processed in {duration:.2f} seconds.")
            logging.info(f"AI Response: {answer}")
            print("\nAI Response:\n", answer)


if __name__ == "__main__":
    mode_input = input("Enter debug mode? (yes/no): ").strip().lower()
    debug_mode = True if mode_input == "yes" else False
    interface = HVLdb5DebugInterface(debug_mode=debug_mode)
    interface.run()
