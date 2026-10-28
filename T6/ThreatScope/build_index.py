from database import init_db
from pipeline import run_embeddings_and_clusters

if __name__ == "__main__":
    init_db()
    run_embeddings_and_clusters()