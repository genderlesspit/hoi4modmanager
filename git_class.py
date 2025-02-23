from git import Repo
import os

# Set the base directory to the PyCharm virtual environment
BASE_DIR = os.path.abspath(".venv")

class GitHubManager:
    @staticmethod
    def retrieve(url, new_dir):
        """Retrieve a GitHub repository and store it inside the venv directory."""
        repo_path = os.path.join(BASE_DIR, new_dir)  # Create full repo path

        if os.path.exists(repo_path):
            print(f"⚠ Directory '{repo_path}' already exists. Choose another name or delete the existing folder.")
            return

        try:
            print(f"🔹 Cloning {url} into {repo_path}...")
            Repo.clone_from(url, repo_path)
            print("✅ Repository successfully cloned!")
        except Exception as e:
            print(f"⚠ Error cloning repository: {e}")

# Example usage
if __name__ == "__main__":
    github_url = input("Enter GitHub repo URL: ")
    repo_name = input("Enter folder name for the repo: ")
    GitHubManager.retrieve(github_url, repo_name)
