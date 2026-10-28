import numpy as np
from sklearn.ensemble import IsolationForest
from sklearn.decomposition import PCA
from features import build_features

def detect_anomalies(reports, emb=None, contamination=0.08):
    if not reports:
        return np.array([])
    df = build_features(reports)
    X = df.values
    if emb is not None and len(emb) == len(df) and emb.size:
        e = PCA(n_components=min(16, emb.shape[1]), random_state=42).fit_transform(emb)
        X = np.hstack([X, e])
    if len(X) < 10:
        return np.zeros(len(X))
    iso = IsolationForest(contamination=contamination, random_state=42)
    iso.fit(X)
    return -iso.score_samples(X)  # higher = more anomalous