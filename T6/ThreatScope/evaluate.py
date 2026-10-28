from database import init_db, fetch_all_reports
from preprocessing import enrich
from classifier import load_baseline, predict
from embeddings import search
from sklearn.metrics import classification_report, accuracy_score

def eval_classification():
    reports = fetch_all_reports()
    pipe = load_baseline()
    if not reports or pipe is None:
        print("Skipping classification eval: missing reports or model.")
        return
    texts = [((r.get("title") or "") + ". " + (r.get("text") or "")) for r in reports]
    y_true = [enrich(dict(r))["threat_type"] for r in reports]
    preds = predict(pipe, texts)
    y_pred = [p[0] for p in preds]
    print("Accuracy:", round(accuracy_score(y_true, y_pred), 4))
    print(classification_report(y_true, y_pred, zero_division=0))

def eval_retrieval(k=5):
    reports = fetch_all_reports()
    if len(reports) < 10:
        print("Skipping retrieval eval: too few reports.")
        return
    hits, mrr = 0, 0.0
    N = min(30, len(reports))
    for r in reports[:N]:
        q = r.get("title", "")
        results = search(q, top_k=k)
        ids = [x["id"] for x in results]
        if r["id"] in ids:
            rank = ids.index(r["id"]) + 1
            hits += 1
            mrr += 1.0 / rank
    print(f"Hit@{k}: {hits / N:.3f} | MRR: {mrr / N:.3f}")

if __name__ == "__main__":
    init_db()
    eval_classification()
    eval_retrieval()