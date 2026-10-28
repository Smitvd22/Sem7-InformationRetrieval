from database import init_db, fetch_all_reports, update_many
from preprocessing import enrich
from classifier import load_baseline, predict
from embeddings import encode, save_index
from clustering import cluster_embeddings
from anomaly import detect_anomalies

def run_enrichment():
    reports = fetch_all_reports()
    updates = []
    for r in reports:
        if r.get("threat_type") and r.get("severity"):
            continue
        rec = enrich(dict(r))
        updates.append((r["id"], {
            "cve_ids": rec["cve_ids"],
            "attack_techniques": rec["attack_techniques"],
            "cvss": rec["cvss"],
            "threat_type": rec["threat_type"],
            "severity": rec["severity"],
        }))
    update_many(updates)
    print(f"[pipeline] enriched {len(updates)} reports")
    return len(updates)

def run_classification():
    pipe = load_baseline()
    if pipe is None:
        print("[pipeline] no classifier found; run train_classifier.py first")
        return 0
    reports = fetch_all_reports()
    texts = [((r.get("title") or "") + ". " + (r.get("text") or "")) for r in reports]
    preds = predict(pipe, texts)
    updates = []
    for r, (label, conf) in zip(reports, preds):
        updates.append((r["id"], {"threat_type": label, "confidence": conf}))
    update_many(updates)
    print(f"[pipeline] classified {len(updates)} reports")
    return len(updates)

def run_embeddings_and_clusters(n_clusters=12):
    reports = fetch_all_reports()
    if not reports:
        print("[pipeline] no reports")
        return
    texts = [((r.get("title") or "") + ". " + (r.get("text") or "")) for r in reports]
    emb = encode(texts)
    ids = [r["id"] for r in reports]
    save_index(emb, ids)
    clusters = cluster_embeddings(emb, n_clusters=n_clusters)
    scores = detect_anomalies(reports, emb=emb)
    updates = []
    for r, c, s in zip(reports, clusters, scores):
        updates.append((r["id"], {"cluster_id": int(c), "anomaly_score": float(s)}))
    update_many(updates)
    print(f"[pipeline] indexed {len(ids)} reports into {len(set(clusters))} clusters")

def run_full_pipeline():
    init_db()
    run_enrichment()
    run_classification()
    run_embeddings_and_clusters()