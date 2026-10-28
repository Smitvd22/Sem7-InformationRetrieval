import os
import pickle
import numpy as np
from config import INDEX_DIR, MODEL_DIR, EMBEDDING_MODEL

# ---------------------------------------------------------------------------
# Pipeline resolution
#  1. Try SentenceTransformer (needs torch).
#  2. Fall back to a fitted TF-IDF + SVD pipeline persisted to disk.
# The pipeline is fit ONCE (during setup/indexing) and reused for queries.
# ---------------------------------------------------------------------------

_pipeline = None
_pipeline_path = MODEL_DIR / "tfidf_svd.joblib"


def _fit_tfidf_svd(texts, target_dim=384):
    """Fit TF-IDF + SVD on the corpus and persist it."""
    from sklearn.feature_extraction.text import TfidfVectorizer
    from sklearn.decomposition import TruncatedSVD
    import joblib

    vec = TfidfVectorizer(max_features=5000, stop_words="english", ngram_range=(1, 2))
    X = vec.fit_transform(texts)

    n_features = X.shape[1]
    n_samples = X.shape[0]
    # SVD needs n_components <= min(n_samples, n_features) - 1
    n_comp = max(2, min(target_dim, n_features - 1, n_samples - 1))

    svd = TruncatedSVD(n_components=n_comp, random_state=42)
    svd.fit(X)

    bundle = {"vec": vec, "svd": svd, "dim": n_comp}
    joblib.dump(bundle, _pipeline_path)
    return bundle


def get_pipeline():
    """Return a tuple (kind, payload). kind is 'st', 'tfidf', or 'tfidf-unfitted'."""
    global _pipeline
    if _pipeline is not None:
        return _pipeline

    # 1) Try SentenceTransformer
    try:
        from sentence_transformers import SentenceTransformer
        _pipeline = ("st", SentenceTransformer(EMBEDDING_MODEL))
        print(f"[embeddings] Using SentenceTransformer ({EMBEDDING_MODEL})")
        return _pipeline
    except Exception as e:
        print(f"[embeddings] SentenceTransformer unavailable -> {e}")
        print("[embeddings] Falling back to TF-IDF + SVD")

    # 2) Try loading a previously fitted pipeline
    if _pipeline_path.exists():
        try:
            import joblib
            bundle = joblib.load(_pipeline_path)
            _pipeline = ("tfidf", bundle)
            print(f"[embeddings] Loaded cached TF-IDF/SVD (dim={bundle.get('dim', '?')})")
            return _pipeline
        except Exception as e:
            print(f"[embeddings] Could not load cached TF-IDF/SVD: {e}")

    _pipeline = ("tfidf-unfitted", None)
    return _pipeline


def encode(texts, batch_size=32, normalize=True):
    """Encode texts. Fits the fallback pipeline on first call with a real corpus."""
    global _pipeline

    if not texts:
        # Return an empty array with a safe dimension placeholder.
        return np.zeros((0, 384), dtype="float32")

    kind, model = get_pipeline()

    if kind == "st":
        emb = model.encode(
            texts,
            batch_size=batch_size,
            show_progress_bar=False,
            convert_to_numpy=True,
        ).astype("float32")
    else:
        # TF-IDF fallback
        if kind == "tfidf-unfitted":
            if len(texts) < 3:
                raise RuntimeError(
                    "TF-IDF/SVD pipeline is not fitted yet. "
                    "Run `python setup_demo.py` or `python build_index.py` first."
                )
            bundle = _fit_tfidf_svd(texts)
            _pipeline = ("tfidf", bundle)
            print(f"[embeddings] Fitted TF-IDF/SVD (dim={bundle['dim']})")
        else:
            bundle = model

        vec, svd = bundle["vec"], bundle["svd"]
        X = vec.transform(texts)
        emb = svd.transform(X).astype("float32")

    if normalize:
        norms = np.linalg.norm(emb, axis=1, keepdims=True) + 1e-9
        emb = emb / norms

    return emb


# ---------------------------------------------------------------------------
# Index persistence
# ---------------------------------------------------------------------------

def save_index(emb, ids, path_prefix="reports"):
    np.save(INDEX_DIR / f"{path_prefix}_emb.npy", emb)
    with open(INDEX_DIR / f"{path_prefix}_ids.pkl", "wb") as f:
        pickle.dump(ids, f)
    try:
        import faiss
        index = faiss.IndexFlatIP(emb.shape[1])
        index.add(emb)
        faiss.write_index(index, str(INDEX_DIR / f"{path_prefix}.faiss"))
    except Exception as e:
        print("[embeddings] FAISS not saved:", e)


def load_index(path_prefix="reports"):
    emb_path = INDEX_DIR / f"{path_prefix}_emb.npy"
    ids_path = INDEX_DIR / f"{path_prefix}_ids.pkl"
    if not emb_path.exists() or not ids_path.exists():
        return None, None, None
    emb = np.load(emb_path)
    with open(ids_path, "rb") as f:
        ids = pickle.load(f)
    index = None
    faiss_path = INDEX_DIR / f"{path_prefix}.faiss"
    if faiss_path.exists():
        try:
            import faiss
            index = faiss.read_index(str(faiss_path))
        except Exception:
            index = None
    return emb, ids, index


def search(query, top_k=10, path_prefix="reports"):
    emb, ids, index = load_index(path_prefix)
    if emb is None:
        return []

    q = encode([query])[0]

    # Guard against backend switch (e.g., ST index vs TF-IDF query)
    if q.shape[0] != emb.shape[1]:
        print(
            f"[search] Dimension mismatch: query dim={q.shape[0]}, "
            f"index dim={emb.shape[1]}. Delete `index/` and rebuild."
        )
        return []

    if index is not None:
        scores, idxs = index.search(q.reshape(1, -1), min(top_k, len(ids)))
        scores, idxs = scores[0], idxs[0]
    else:
        sims = emb @ q
        idxs = np.argsort(-sims)[:top_k]
        scores = sims[idxs]

    return [
        {"id": int(ids[i]), "score": float(s)}
        for i, s in zip(idxs, scores) if i >= 0
    ]