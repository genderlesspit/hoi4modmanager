import sqlite3
import os
import ollama

def query_ai(prompt):
    """Query Ollama AI (Llama 3) as a free alternative to OpenAI."""
    try:
        print("🔹 Sending request to Ollama...")
        response = ollama.chat(model="llama3", messages=[{"role": "user", "content": prompt}])
        print("🔹 Response received!")
        return response["message"]["content"].strip()
    except Exception as e:
        return f"⚠ Ollama Error: {e}"


def initialize_databases():
    """Create necessary databases if they don't exist and optimize them."""
    conn = sqlite3.connect("hoi4_feedback.sqlite", check_same_thread=False)
    cursor = conn.cursor()

    # Enable Write-Ahead Logging for faster writes
    cursor.execute("PRAGMA journal_mode=WAL;")

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS amended_logic (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        original_text TEXT,
        amended_text TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS user_feedback (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        response TEXT,
        feedback TEXT
    )
    """)

    cursor.execute("""
    CREATE TABLE IF NOT EXISTS prompt_log (
        id INTEGER PRIMARY KEY AUTOINCREMENT,
        prompt TEXT,
        response TEXT,
        timestamp DATETIME DEFAULT CURRENT_TIMESTAMP
    )
    """)

    # Create indexes for faster queries
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_embeddings_text ON amended_logic(original_text);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_feedback_response ON user_feedback(response);")
    cursor.execute("CREATE INDEX IF NOT EXISTS idx_prompt_log ON prompt_log(prompt);")

    conn.commit()
    return conn, cursor  # Return persistent connection


def store_feedback(cursor, conn, response, feedback, table):
    """Store user feedback efficiently using batch inserts."""
    data = [(response, feedback)]
    cursor.executemany(f"INSERT INTO {table} (response, feedback) VALUES (?, ?)", data)
    conn.commit()


def log_prompt(cursor, conn, prompt, response):
    """Log all AI interactions for persistent memory."""
    cursor.execute("INSERT INTO prompt_log (prompt, response) VALUES (?, ?)", (prompt, response))
    conn.commit()


def test_ai(cursor, conn):
    """Test function for free chat with the AI and review responses."""
    while True:
        user_prompt = input("You: ")
        if user_prompt.lower() in ["exit", "quit"]:
            print("🔹 Exiting AI chat.")
            break

        response = query_ai(user_prompt)
        log_prompt(cursor, conn, user_prompt, response)

        print("\n==== AI Response ====")
        print(response)

        print("\nReview this response:")
        print("1. Amend logic")
        print("2. Approve response")
        print("3. Reject response")
        choice = input("Enter choice (1/2/3): ")

        if choice == "1":
            amended_text = input("Enter your amendment: ")
            store_feedback(cursor, conn, response, amended_text, "amended_logic")
            print("✅ Amended logic stored.")
        elif choice == "2":
            store_feedback(cursor, conn, response, "Approved", "user_feedback")
            print("✅ Response approved.")
        elif choice == "3":
            store_feedback(cursor, conn, response, "Rejected", "user_feedback")
            print("✅ Response rejected.")
        else:
            print("⚠ Invalid choice. No action taken.")


if __name__ == "__main__":
    conn, cursor = initialize_databases()
    test_ai(cursor, conn)
