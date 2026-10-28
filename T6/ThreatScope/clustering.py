import numpy as np
from sklearn.cluster import KMeans
from sklearn.decomposition import NMF, PCA
from sklearn.manifold import TSNE
from sklearn.feature_extraction.text import TfidfVectorizer

def cluster_embeddings(emb, n_clusters=12):
    if len(emb) < 2:
        return np.zeros(len(emb), dtype=int)
    k = min(n_clusters, max(2, len(emb) // 3))
    km = KMeans(n_clusters=k, random_state=42, n_init="auto")
    return km.fit_predict(emb)

def topic_model(texts, n_topics=8, n_words=8):
    vec = TfidfVectorizer(max_features=5000, stop_words="english")
    X = vec.fit_transform(texts)
    if X.shape[0] < 3:
        return [], X
    k = min(n_topics, max(2, X.shape[0] - 1))
    nmf = NMF(n_components=k, random_state=42, init="nndsvd", max_iter=300)
    W = nmf.fit_transform(X)
    terms = np.array(vec.get_feature_names_out())
    topics = []
    for comp in nmf.components_:
        topics.append(list(terms[np.argsort(-comp)[:n_words]]))
    return topics, W

def reduce_dim(emb, method="pca", n_components=2):
    if len(emb) < 3:
        if emb.shape[1] >= n_components:
            return emb[:, :n_components]
        return np.zeros((len(emb), n_components))
    if method == "tsne" and len(emb) > 10:
        return TSNE(
            n_components=n_components,
            perplexity=min(30, len(emb) - 1),
            random_state=42,
        ).fit_transform(emb)
    return PCA(n_components=n_components, random_state=42).fit_transform(emb)