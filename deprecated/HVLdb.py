"""
==================================================
🔹 HOI4 Modding AI - Vector Database with LLM 🔹 v1
==================================================

This script builds an AI-powered assistant for Hearts of Iron IV (HOI4) modding.
It leverages **ChromaDB** for vector similarity search and **Llama3 / Falcon-7B**
for intelligent query responses.

==================================================
📌 CLASSES:
==================================================

✅ class `HOI4VectorDB`
    - Manages the ChromaDB vector database.
    - Stores, retrieves, and queries modding knowledge.

✅ class `HOI4ModdingAI`
    - Retrieves context from the vector database.
    - Generates accurate responses using Falcon-7B or Llama3.
    - Ensures AI doesn't "hallucinate" incorrect answers.

==================================================
📌 FUNCTIONS (HOI4VectorDB):
==================================================

🔹 `__init__(db_name="hoi4_vector_db", embed_model="sentence-transformers/all-MiniLM-L6-v2")`
    - Initializes the ChromaDB client and embedding model.
    - ✅ **Variables:**
        - `db_name (str)`: Name of the ChromaDB database.
        - `embed_model (str)`: SentenceTransformer model for text embedding.

🔹 `embed_text(text: str) -> list`
    - Converts input text into a vector embedding.
    - ✅ **Variables:**
        - `text (str)`: The input text to be embedded.
    - ✅ **Returns:**
        - `list`: A numerical vector representing the text.

🔹 `add_data(texts: list, metadata: dict = None) -> None`
    - Adds modding knowledge to ChromaDB.
    - ✅ **Variables:**
        - `texts (list)`: A list of text documents to store in the database.
        - `metadata (dict, optional)`: Additional metadata for each document.
    - ✅ **Returns:**
        - `None`

🔹 `query(text: str, top_k: int = 3, similarity_threshold: float = 0.75) -> tuple`
    - Searches ChromaDB for relevant modding data.
    - Uses cosine similarity to find the best match.
    - If similarity is **too low**, alerts the user instead of guessing.
    - ✅ **Variables:**
        - `text (str)`: The query string from the user.
        - `top_k (int)`: Number of closest matches to retrieve.
        - `similarity_threshold (float)`: Minimum similarity score required.
    - ✅ **Returns:**
        - `tuple`: `(retrieved_text (str) or None, similarity_score (float))`

==================================================
📌 FUNCTIONS (HOI4ModdingAI):
==================================================

🔹 `__init__(vector_db: HOI4VectorDB, use_falcon: bool = True)`
    - Initializes the AI with the selected model.
    - ✅ **Variables:**
        - `vector_db (HOI4VectorDB)`: The vector database manager.
        - `use_falcon (bool)`: If `True`, uses Falcon-7B; otherwise, Llama3.

🔹 `generate_answer(prompt: str) -> str`
    - Uses Falcon-7B or Llama3 to generate AI responses.
    - ✅ **Variables:**
        - `prompt (str)`: The input text prompt to pass to the AI.
    - ✅ **Returns:**
        - `str`: The generated response from the model.

🔹 `query_ai(user_query: str) -> str`
    - Retrieves **validated** context from the vector database.
    - Ensures similarity score is high before passing data to the AI.
    - Logs performance & prevents hallucinations.
    - ✅ **Variables:**
        - `user_query (str)`: The user’s input question.
    - ✅ **Returns:**
        - `str`: Either a valid AI response or a warning message.

🔹 `chat() -> None`
    - Interactive chat session for asking modding-related questions.
    - Uses `query_ai()` to answer user queries.
    - ✅ **Variables:**
        - None
    - ✅ **Returns:**
        - `None`

==================================================
📌 HOW TO RUN:
==================================================

1️⃣ Install dependencies (if not installed):
    ```python
    pip install chromadb sentence-transformers torch transformers ollama
    ```

2️⃣ Run the AI:
    ```python
    if __name__ == "__main__":
        db_manager = HOI4VectorDB()
        db_manager.add_data([
            "A focus tree is defined by using a focus_tree = { ... } block.",
            "What are the effects of stability on HOI4 gameplay?",
            "How can I add an event that triggers after 100 days?"
        ])

        hoi4_ai = HOI4ModdingAI(db_manager, use_falcon=False)  # Use Llama3 by default
        hoi4_ai.chat()
    ```

3️⃣ Start asking questions about HOI4 modding!

==================================================
📌 LOGGING & DEBUGGING:
==================================================

✅ **All logs are stored in:** `hoi4_ai_debug.log`
✅ **Tracks similarity scores, performance, and errors.**
✅ **Ensures AI does not generate incorrect responses.**
"""

import chromadb
import logging
import time
import numpy as np
from sentence_transformers import SentenceTransformer
import torch
import transformers
import ollama
from transformers import AutoTokenizer, AutoModelForCausalLM

# ✅ Set up logging
logging.basicConfig(filename="hoi4_ai_debug.log", level=logging.DEBUG,
                    format="%(asctime)s - %(levelname)s - %(message)s")

class HOI4VectorDB:
    """Handles vector embeddings and queries for HOI4 modding knowledge."""

    def __init__(self, db_name="hoi4_vector_db", embed_model="sentence-transformers/all-MiniLM-L6-v2"):
        """Initialize the vector database and embedding model."""
        self.client = chromadb.PersistentClient(path=f"./{db_name}")
        self.collection = self.client.get_or_create_collection(name=db_name)
        self.embed_model = SentenceTransformer(embed_model)

    def embed_text(self, text):
        """Generate a vector embedding for the input text."""
        return self.embed_model.encode(text).tolist()

    def add_data(self, texts, metadata=None):
        """Add documents and their embeddings to the database."""
        embeddings = [self.embed_text(text) for text in texts]
        ids = [f"id_{i}" for i in range(len(texts))]

        self.collection.add(embeddings=embeddings, documents=texts, metadatas=metadata, ids=ids)

        print(f"✅ {len(texts)} entries added to ChromaDB")
        print(f"🔍 Checking stored embeddings count: {self.collection.count()}")  # Debugging

    def query(self, text, top_k=3, similarity_threshold=0.75):
        """Retrieve the most relevant context based on similarity."""
        start_time = time.time()

        # 🔍 Ensure there are stored embeddings before querying
        if self.collection.count() == 0:
            logging.warning("⚠️ No stored embeddings found! Add data before querying.")
            return None, 0.0

        query_embedding = np.array(self.embed_text(text))
        results = self.collection.query(query_embeddings=[query_embedding.tolist()], n_results=top_k)

        end_time = time.time()

        if not results["documents"]:
            logging.warning(f"⚠️ No relevant context found for query: '{text}'")
            return None, 0.0

        retrieved_docs = results["documents"][0]
        if "embeddings" not in results or not results["embeddings"]:
            logging.error(f"🚨 Embeddings retrieval failed! Results: {results}")
            return None, 0.0

        retrieved_embeddings = np.array(results["embeddings"][0])  # Get stored embeddings

        # ✅ Compute similarity scores
        similarities = np.dot(query_embedding, retrieved_embeddings.T) / (
                np.linalg.norm(query_embedding) * np.linalg.norm(retrieved_embeddings, axis=1)
        )
        best_similarity = np.max(similarities)

        logging.debug(
            f"Query: '{text}' | Best Similarity Score: {best_similarity:.4f} | Retrieved in {end_time - start_time:.4f} sec")

        if best_similarity < similarity_threshold:
            return None, best_similarity

        return "\n".join(retrieved_docs), best_similarity

class HOI4ModdingAI:
    """Retrieves HOI4 modding knowledge and generates responses using Falcon-7B or Llama3."""

    def __init__(self, vector_db, use_falcon=True):
        """Initialize the AI model and vector database."""
        self.vector_db = vector_db
        self.use_falcon = use_falcon
        self.total_queries = 0
        self.valid_queries = 0

        if use_falcon:
            self.model_id = "tiiuae/falcon-7b-instruct"
            self.device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
            self.tokenizer = AutoTokenizer.from_pretrained(self.model_id)
            self.pipeline = transformers.pipeline("text-generation",
                                                  model=self.model_id,
                                                  tokenizer=self.tokenizer,
                                                  torch_dtype=torch.bfloat16,
                                                  trust_remote_code=True,
                                                  device_map="auto")
        else:
            self.model_id = "llama3"

    def generate_answer(self, prompt):
        """Generate an AI response using Falcon-7B or Llama3."""
        start_time = time.time()

        if self.use_falcon:
            sequences = self.pipeline(prompt, max_length=200, do_sample=True, top_k=10, num_return_sequences=1)
            response = sequences[0]['generated_text']
        else:
            response = ollama.chat(model=self.model_id, messages=[{"role": "user", "content": prompt}])["message"][
                "content"].strip()

        end_time = time.time()
        logging.debug(f"Generated response in {end_time - start_time:.4f} sec | Prompt Length: {len(prompt)} chars")

        return response

    def query_ai(self, user_query):
        """Retrieve vector context and generate a response using AI."""
        logging.info(f"User Query: {user_query}")
        start_time = time.time()

        context, similarity_score = self.vector_db.query(user_query, top_k=3, similarity_threshold=0.75)
        self.total_queries += 1  # Track total queries

        # ✅ If context is weak, alert the user instead of generating a bad answer
        if context is None:
            logging.warning(f"Low confidence query: '{user_query}' | Similarity Score: {similarity_score:.4f}")
            accuracy = self.valid_queries / self.total_queries if self.total_queries > 0 else 0
            logging.info(f"Performance Score: {self.valid_queries}/{self.total_queries} ({accuracy:.2%})")
            return f"⚠️ The database does not have enough information to answer this accurately. Try rephrasing or adding more context. (Similarity Score: {similarity_score:.4f})"

        self.valid_queries += 1  # Track valid queries
        accuracy = self.valid_queries / self.total_queries if self.total_queries > 0 else 0
        logging.info(
            f"Valid query accepted | Similarity Score: {similarity_score:.4f} | Performance Score: {self.valid_queries}/{self.total_queries} ({accuracy:.2%})")

        full_prompt = f"Relevant Context:\n{context}\n\nUser Query:\n{user_query}"
        response = self.generate_answer(full_prompt)

        end_time = time.time()
        logging.info(f"Total query time: {end_time - start_time:.4f} sec")

        return response

    def chat(self):
        """Interactive chat for HOI4 modding queries."""
        print("\n=== HOI4 Modding AI ===")
        while True:
            user_input = input("\nAsk HOI4 AI: ")
            if user_input.lower() in ["exit", "quit"]:
                break
            response = self.query_ai(user_input)
            print("\nAI Response:", response)


# ✅ Example Usage
if __name__ == "__main__":
    db_manager = HOI4VectorDB()
    db_manager.add_data([
        "A focus tree is defined by using a focus_tree = { ... } block.",
        "What are the effects of stability on HOI4 gameplay?",
        "How can I add an event that triggers after 100 days?"
    ])

    hoi4_ai = HOI4ModdingAI(db_manager, use_falcon=False)  # Use Llama3 by default
    hoi4_ai.chat()
