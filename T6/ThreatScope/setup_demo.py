from database import init_db, fetch_all_reports
from ingestion import load_sample_data
from pipeline import run_enrichment, run_classification, run_embeddings_and_clusters
from preprocessing import enrich
from classifier import train_baseline

def train_quick_baseline():
    reports = fetch_all_reports()
    texts, labels = [], []
    for r in reports:
        rec = enrich(dict(r))
        if rec["threat_type"] == "other":
            continue
        texts.append((r.get("title") or "") + ". " + (r.get("text") or ""))
        labels.append(rec["threat_type"])
    if len(set(labels)) >= 2:
        train_baseline(texts, labels)
        print("[setup] baseline classifier trained")
    else:
        print("[setup] not enough labels; skipping baseline")

if __name__ == "__main__":
    print("Initializing database...")
    init_db()

    print("Loading sample data...")
    load_sample_data("data/sample_reports.csv")

    print("Enriching...")
    run_enrichment()

    print("Training baseline...")
    train_quick_baseline()

    print("Classifying...")
    run_classification()

    print("Building embeddings, clusters, anomalies...")
    run_embeddings_and_clusters()

    print("\nDemo setup complete. Run: streamlit run app.py")