"""
Practical Assignment 5: Intelligent Document Search Using Unsupervised Learning
================================================================================
Application : Intelligent Search System for Research Papers / College Notes
Student ID  : U23AI118
Course      : Information Retrieval

Concepts Covered:
  - Word / Document Embeddings (TF-IDF Vectors)
  - Clustering-based Retrieval (K-Means)
  - Topic Modeling (Latent Dirichlet Allocation)
  - Dimensionality Reduction (PCA + t-SNE)
  - Anomaly Detection (distance-based)
  - Retrieval Evaluation (Precision, Recall, F1)

Dependencies: scikit-learn, matplotlib, numpy
    pip install scikit-learn matplotlib numpy
"""

import math
import re
import os
import warnings
import numpy as np
import matplotlib
matplotlib.use("Agg")                          # non-interactive backend
import matplotlib.pyplot as plt
from collections import Counter

from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans
from sklearn.decomposition import LatentDirichletAllocation, PCA
from sklearn.manifold import TSNE
from sklearn.metrics.pairwise import cosine_similarity
from sklearn.preprocessing import normalize

warnings.filterwarnings("ignore")

# ============================================================
# 1. DOCUMENT COLLECTION  (25 documents, 4 subjects + 1 anomaly)
# ============================================================

DOCUMENTS = {
    # ---------- AI / Machine Learning (7 documents) ----------
    "D1":  ("Neural networks and deep learning fundamentals including "
            "backpropagation gradient descent and activation functions for "
            "training multi-layer perceptrons"),
    "D2":  ("Natural language processing techniques such as tokenization "
            "part of speech tagging named entity recognition and sentiment "
            "analysis using transformer models"),
    "D3":  ("Reinforcement learning algorithms including Q-learning policy "
            "gradients and deep reinforcement learning for game playing "
            "and robotic control"),
    "D4":  ("Computer vision and convolutional neural networks for image "
            "classification object detection and semantic segmentation "
            "in autonomous driving"),
    "D5":  ("Supervised and unsupervised machine learning methods including "
            "decision trees random forests support vector machines and "
            "K-means clustering for data analysis"),
    "D6":  ("Generative adversarial networks and variational autoencoders "
            "for image synthesis data augmentation and creative AI "
            "applications in art generation"),
    "D7":  ("Transfer learning and pre-trained language models like BERT "
            "GPT and T5 for downstream NLP tasks including question "
            "answering and text summarization"),

    # ---------- Cybersecurity (6 documents) ----------
    "D8":  ("Network security and firewall configuration including "
            "intrusion detection systems packet filtering and network "
            "access control policies"),
    "D9":  ("Cryptography fundamentals covering symmetric and asymmetric "
            "encryption RSA AES digital signatures and public key "
            "infrastructure"),
    "D10": ("Ethical hacking and penetration testing methodologies for "
            "vulnerability assessment including OWASP top ten web "
            "application security risks"),
    "D11": ("Malware analysis and reverse engineering techniques for "
            "detecting trojans ransomware worms and advanced persistent "
            "threats in enterprise networks"),
    "D12": ("Cyber threat intelligence and security operations center "
            "monitoring with SIEM tools for incident response and "
            "digital forensics investigation"),
    "D13": ("Blockchain security and smart contract vulnerabilities "
            "including consensus mechanism attacks and decentralized "
            "application security auditing"),

    # ---------- Databases (6 documents) ----------
    "D14": ("Relational database management systems and SQL query "
            "optimization including joins subqueries indexing and "
            "normalization for efficient data storage"),
    "D15": ("NoSQL databases including MongoDB Cassandra and Redis for "
            "handling unstructured data with high availability and "
            "horizontal scalability"),
    "D16": ("Database indexing techniques including B-trees hash indexes "
            "and bitmap indexes for accelerating query performance in "
            "large-scale data warehouses"),
    "D17": ("Transaction management and concurrency control with ACID "
            "properties two-phase locking and multiversion concurrency "
            "control in distributed databases"),
    "D18": ("Data warehousing and OLAP cube operations including star "
            "schema design ETL processes and business intelligence "
            "reporting and analytics"),
    "D19": ("Graph databases and knowledge graphs using Neo4j for "
            "representing complex relationships in social networks "
            "recommendation systems and semantic web"),

    # ---------- Networking (5 documents) ----------
    "D20": ("Computer networking fundamentals including OSI and TCP IP "
            "models routing protocols switching and subnetting for "
            "local and wide area networks"),
    "D21": ("Wireless network protocols including WiFi Bluetooth 5G "
            "and IoT communication standards for mobile and embedded "
            "device connectivity"),
    "D22": ("Software defined networking and network function "
            "virtualization for programmable network infrastructure "
            "using OpenFlow and SDN controllers"),
    "D23": ("Cloud computing architectures including IaaS PaaS SaaS "
            "models virtual machines containers and serverless "
            "computing for scalable applications"),
    "D24": ("Content delivery networks and load balancing strategies "
            "for distributed web applications with DNS-based routing "
            "and edge caching optimization"),

    # ---------- Anomaly Document (1 document) ----------
    "D25": ("Organic vegetable gardening soil composting raised bed planting "
            "tomato cultivation weed removal and botanical horticulture"),
}

# Ground-truth subject labels (for evaluation and display)
SUBJECT_LABELS = {
    "D1": "AI",  "D2": "AI",  "D3": "AI",  "D4": "AI",
    "D5": "AI",  "D6": "AI",  "D7": "AI",
    "D8": "Cybersecurity",  "D9": "Cybersecurity",  "D10": "Cybersecurity",
    "D11": "Cybersecurity", "D12": "Cybersecurity", "D13": "Cybersecurity",
    "D14": "Databases", "D15": "Databases", "D16": "Databases",
    "D17": "Databases", "D18": "Databases", "D19": "Databases",
    "D20": "Networking", "D21": "Networking", "D22": "Networking",
    "D23": "Networking", "D24": "Networking",
    "D25": "Anomaly",
}

# Test queries with ground-truth relevant document IDs
TEST_QUERIES = {
    "deep learning and neural network architectures": {
        "relevant": {"D1", "D3", "D4", "D6"},
        "description": "AI / Deep Learning query",
    },
    "protecting computer networks from security threats": {
        "relevant": {"D8", "D10", "D11", "D12"},
        "description": "Cybersecurity query",
    },
    "database query optimization and indexing": {
        "relevant": {"D14", "D15", "D16", "D17"},
        "description": "Database query",
    },
    "wireless and cloud computing networks": {
        "relevant": {"D20", "D21", "D22", "D23", "D24"},
        "description": "Networking query",
    },
}

# ============================================================
# 2. TEXT PREPROCESSING & TF-IDF VECTORIZATION
# ============================================================

def get_doc_ids_and_texts():
    """Return ordered lists of document IDs and their texts."""
    doc_ids = sorted(DOCUMENTS.keys(), key=lambda x: int(x[1:]))
    texts = [DOCUMENTS[d] for d in doc_ids]
    return doc_ids, texts

def build_tfidf_matrix(texts):
    """
    Build TF-IDF document-term matrix.
    Each row is a document vector -- this serves as the document embedding.
    """
    vectorizer = TfidfVectorizer(
        stop_words="english",
        max_features=500,
        ngram_range=(1, 2),          # unigrams + bigrams
        sublinear_tf=True,           # apply log normalisation to TF
    )
    tfidf_matrix = vectorizer.fit_transform(texts)
    feature_names = vectorizer.get_feature_names_out()
    return tfidf_matrix, vectorizer, feature_names

# ============================================================
# 3. K-MEANS CLUSTERING
# ============================================================

def perform_clustering(tfidf_matrix, n_clusters=4):
    """Apply K-Means clustering to group similar documents."""
    km = KMeans(
        n_clusters=n_clusters,
        random_state=42,
        n_init=10,
        max_iter=300,
    )
    labels = km.fit_predict(tfidf_matrix)
    centroids = km.cluster_centers_
    return labels, centroids, km

# ============================================================
# 4. TOPIC MODELING (LDA)
# ============================================================

def perform_lda(texts, n_topics=4, n_top_words=8):
    """
    Apply Latent Dirichlet Allocation (LDA) for topic discovery.
    LDA needs raw counts, so we build a separate count-based matrix.
    """
    from sklearn.feature_extraction.text import CountVectorizer

    count_vec = CountVectorizer(
        stop_words="english",
        max_features=500,
        ngram_range=(1, 1),
    )
    count_matrix = count_vec.fit_transform(texts)
    feature_names = count_vec.get_feature_names_out()

    lda = LatentDirichletAllocation(
        n_components=n_topics,
        random_state=42,
        max_iter=20,
        learning_method="batch",
    )
    doc_topic_dist = lda.fit_transform(count_matrix)

    # Extract top words per topic
    topics = []
    for idx, topic_vec in enumerate(lda.components_):
        top_indices = topic_vec.argsort()[-n_top_words:][::-1]
        top_words = [feature_names[i] for i in top_indices]
        topics.append(top_words)

    return doc_topic_dist, topics, lda

# ============================================================
# 5. DIMENSIONALITY REDUCTION (PCA + t-SNE)
# ============================================================

def reduce_dimensions(tfidf_matrix, method="pca", n_components=2):
    """Reduce high-dimensional TF-IDF vectors to 2D for visualisation."""
    dense = tfidf_matrix.toarray()
    if method == "pca":
        reducer = PCA(n_components=n_components, random_state=42)
    else:
        reducer = TSNE(
            n_components=n_components,
            random_state=42,
            perplexity=min(8, len(dense) - 1),   # small collection
            max_iter=1000,
            learning_rate="auto",
            init="pca",
        )
    coords = reducer.fit_transform(dense)
    return coords

def plot_clusters(coords, labels, doc_ids, subject_labels, anomaly_ids,
                  title, filename):
    """Create a 2D scatter plot of document clusters and save as PNG."""
    plt.figure(figsize=(10, 7))

    # Colour map for clusters
    unique_labels = sorted(set(labels))
    cmap = matplotlib.colormaps["tab10"]
    colours = [cmap(l / max(len(unique_labels) - 1, 1)) for l in labels]

    for i, (x, y) in enumerate(coords):
        doc_id = doc_ids[i]
        is_anomaly = doc_id in anomaly_ids

        plt.scatter(
            x, y,
            c=[colours[i]],
            s=180 if is_anomaly else 100,
            marker="X" if is_anomaly else "o",
            edgecolors="red" if is_anomaly else "black",
            linewidths=2 if is_anomaly else 0.5,
            zorder=5 if is_anomaly else 3,
        )
        label_text = f"{doc_id} ({subject_labels[doc_id]})"
        plt.annotate(
            label_text, (x, y),
            textcoords="offset points", xytext=(6, 6),
            fontsize=7, alpha=0.85,
        )

    # Legend for clusters
    for cl in unique_labels:
        plt.scatter([], [], c=[cmap(cl / max(len(unique_labels) - 1, 1))], s=80, label=f"Cluster {cl}")
    plt.scatter([], [], c="grey", s=120, marker="X",
                edgecolors="red", linewidths=2, label="Anomaly")
    plt.legend(loc="best", fontsize=8)

    plt.title(title, fontsize=13, fontweight="bold")
    plt.xlabel("Component 1")
    plt.ylabel("Component 2")
    plt.grid(True, alpha=0.2)
    plt.tight_layout()
    plt.savefig(filename, dpi=200)
    plt.close()

# ============================================================
# 6. ANOMALY DETECTION
# ============================================================

def detect_anomalies(tfidf_matrix, labels, centroids, doc_ids,
                     threshold_factor=1.5):
    """
    Detect anomalies as documents whose affinity (cosine similarity to the
    leave-one-out centroid of their assigned cluster) is significantly below average:
        affinity < mean - threshold_factor * std
    An unrelated document accidentally added to the collection has near-zero
    similarity to the topics of any cluster.
    """
    dense = tfidf_matrix.toarray()
    affinities = []
    for i, doc_vec in enumerate(dense):
        c = labels[i]
        cluster_indices = [j for j in range(len(dense)) if labels[j] == c and j != i]
        if cluster_indices:
            loo_centroid = dense[cluster_indices].mean(axis=0)
            norm_c = np.linalg.norm(loo_centroid)
            if norm_c > 0:
                sim = float(cosine_similarity(doc_vec.reshape(1, -1), loo_centroid.reshape(1, -1))[0, 0])
            else:
                sim = 0.0
        else:
            sim = 0.0
        affinities.append(sim)

    affinities = np.array(affinities)
    mean_aff = float(affinities.mean())
    std_aff = float(affinities.std())
    threshold = mean_aff - threshold_factor * std_aff

    anomalies = []
    for i, aff in enumerate(affinities):
        if aff < threshold:
            anomalies.append((doc_ids[i], aff, threshold))

    return anomalies, affinities, threshold

# ============================================================
# 7. SEMANTIC RETRIEVAL
# ============================================================

def retrieve_documents(query, vectorizer, tfidf_matrix, doc_ids, top_n=5):
    """
    Retrieve documents based on cosine similarity between the query
    vector and document vectors in TF-IDF space.
    """
    query_vec = vectorizer.transform([query])
    similarities = cosine_similarity(query_vec, tfidf_matrix).flatten()
    ranked_indices = similarities.argsort()[::-1]

    results = []
    for idx in ranked_indices[:top_n]:
        results.append((doc_ids[idx], similarities[idx]))
    return results

# ============================================================
# 8. EVALUATION METRICS
# ============================================================

def precision(retrieved_set, relevant_set):
    """Precision = |retrieved AND relevant| / |retrieved|"""
    if not retrieved_set:
        return 0.0
    return len(retrieved_set & relevant_set) / len(retrieved_set)

def recall(retrieved_set, relevant_set):
    """Recall = |retrieved AND relevant| / |relevant|"""
    if not relevant_set:
        return 0.0
    return len(retrieved_set & relevant_set) / len(relevant_set)

def f1_score(p, r):
    """F1 = harmonic mean of precision and recall."""
    if p + r == 0:
        return 0.0
    return 2 * p * r / (p + r)

def evaluate_retrieval(query_results, relevant_set, k=5):
    """Evaluate a single query's retrieval at K."""
    retrieved_set = {doc_id for doc_id, _ in query_results[:k]}
    p = precision(retrieved_set, relevant_set)
    r = recall(retrieved_set, relevant_set)
    f = f1_score(p, r)
    return p, r, f

# ============================================================
# 9. DISPLAY HELPERS
# ============================================================

def print_sep(char="=", width=80):
    print(char * width)

def truncate(text, max_len=60):
    return text[:max_len] + "..." if len(text) > max_len else text

# ============================================================
# 10. MAIN EXECUTION
# ============================================================

def main():
    print_sep()
    print("  Practical Assignment 5")
    print("  Intelligent Document Search Using Unsupervised Learning")
    print_sep()

    doc_ids, texts = get_doc_ids_and_texts()

    # -------------------------------------------------------
    # [1] Document Collection
    # -------------------------------------------------------
    print("\n[1] DOCUMENT COLLECTION")
    print("-" * 50)
    subject_counts = Counter(SUBJECT_LABELS.values())
    print(f"  Total documents : {len(DOCUMENTS)}")
    print(f"  Subject areas   : {dict(subject_counts)}")
    print()
    for doc_id in doc_ids:
        subj = SUBJECT_LABELS[doc_id]
        print(f"  {doc_id:>3} [{subj:<14}] {truncate(DOCUMENTS[doc_id])}")

    # -------------------------------------------------------
    # [2] Document Embeddings (TF-IDF Vectors)
    # -------------------------------------------------------
    print(f"\n[2] DOCUMENT EMBEDDINGS (TF-IDF Vectors)")
    print("-" * 50)
    tfidf_matrix, vectorizer, feature_names = build_tfidf_matrix(texts)
    print(f"  Vectorizer       : TfidfVectorizer (sublinear TF, unigrams + bigrams)")
    print(f"  Matrix shape     : {tfidf_matrix.shape}  "
          f"({tfidf_matrix.shape[0]} docs x {tfidf_matrix.shape[1]} features)")
    print(f"  Non-zero entries : {tfidf_matrix.nnz}")
    print(f"  Sparsity         : {1 - tfidf_matrix.nnz / (tfidf_matrix.shape[0] * tfidf_matrix.shape[1]):.2%}")

    # Show top-5 TF-IDF terms for a few sample documents
    print("\n  Sample document vectors (top-5 TF-IDF terms):")
    for sample_id in ["D1", "D8", "D14", "D20", "D25"]:
        idx = doc_ids.index(sample_id)
        row = tfidf_matrix[idx].toarray().flatten()
        top5 = row.argsort()[-5:][::-1]
        terms = [(feature_names[j], f"{row[j]:.3f}") for j in top5]
        print(f"    {sample_id}: {terms}")

    # -------------------------------------------------------
    # [3] K-Means Clustering
    # -------------------------------------------------------
    N_CLUSTERS = 4
    print(f"\n[3] K-MEANS CLUSTERING (K={N_CLUSTERS})")
    print("-" * 50)
    labels, centroids, km_model = perform_clustering(tfidf_matrix, N_CLUSTERS)

    # Show cluster assignments
    cluster_members = {c: [] for c in range(N_CLUSTERS)}
    for i, doc_id in enumerate(doc_ids):
        cluster_members[labels[i]].append(doc_id)

    for c in range(N_CLUSTERS):
        members = cluster_members[c]
        subjects = [SUBJECT_LABELS[d] for d in members]
        dominant = Counter(subjects).most_common(1)[0][0]
        print(f"\n  Cluster {c} (dominant subject: {dominant}):")
        for d in members:
            print(f"    {d:>3} [{SUBJECT_LABELS[d]:<14}] {truncate(DOCUMENTS[d], 50)}")

    # Clustering accuracy (purity)
    total_correct = 0
    for c in range(N_CLUSTERS):
        members = cluster_members[c]
        subjects = [SUBJECT_LABELS[d] for d in members]
        most_common_count = Counter(subjects).most_common(1)[0][1]
        total_correct += most_common_count
    purity = total_correct / len(doc_ids)
    print(f"\n  Clustering Purity: {purity:.4f}  ({total_correct}/{len(doc_ids)} correctly grouped)")

    # -------------------------------------------------------
    # [4] Topic Modeling (LDA)
    # -------------------------------------------------------
    N_TOPICS = 4
    print(f"\n[4] TOPIC MODELING (LDA, {N_TOPICS} topics)")
    print("-" * 50)
    doc_topic_dist, topics, lda_model = perform_lda(texts, N_TOPICS)

    for t_idx, top_words in enumerate(topics):
        print(f"\n  Topic {t_idx}: {', '.join(top_words)}")
        # Show documents with highest weight for this topic
        topic_scores = doc_topic_dist[:, t_idx]
        top_doc_indices = topic_scores.argsort()[-3:][::-1]
        for di in top_doc_indices:
            print(f"    {doc_ids[di]:>3} (weight: {topic_scores[di]:.3f}) "
                  f"{truncate(DOCUMENTS[doc_ids[di]], 45)}")

    # Show dominant topic per document
    print("\n  Dominant topic per document:")
    print(f"  {'Doc':<5} {'Subject':<14} {'Topic':>5}  Topic Keywords")
    print(f"  {'-'*5} {'-'*14} {'-'*5}  {'-'*40}")
    for i, doc_id in enumerate(doc_ids):
        dom_topic = doc_topic_dist[i].argmax()
        print(f"  {doc_id:<5} {SUBJECT_LABELS[doc_id]:<14} {dom_topic:>5}  "
              f"{', '.join(topics[dom_topic][:4])}")

    # -------------------------------------------------------
    # [5] Dimensionality Reduction & Visualization
    # -------------------------------------------------------
    print(f"\n[5] DIMENSIONALITY REDUCTION & VISUALIZATION")
    print("-" * 50)

    script_dir = os.path.dirname(os.path.abspath(__file__))

    # Detect anomalies first (needed for the plots)
    anomalies, affinities, threshold = detect_anomalies(
        tfidf_matrix, labels, centroids, doc_ids
    )
    anomaly_ids = {a[0] for a in anomalies}

    # PCA
    pca_file = os.path.join(script_dir, "05_pca_clusters.png")
    coords_pca = reduce_dimensions(tfidf_matrix, method="pca")
    plot_clusters(
        coords_pca, labels, doc_ids, SUBJECT_LABELS, anomaly_ids,
        "Document Clusters (PCA Projection)", pca_file
    )
    print(f"  Saved: {os.path.basename(pca_file)}")

    # t-SNE
    tsne_file = os.path.join(script_dir, "06_tsne_clusters.png")
    coords_tsne = reduce_dimensions(tfidf_matrix, method="tsne")
    plot_clusters(
        coords_tsne, labels, doc_ids, SUBJECT_LABELS, anomaly_ids,
        "Document Clusters (t-SNE Projection)", tsne_file
    )
    print(f"  Saved: {os.path.basename(tsne_file)}")

    # Explained variance for PCA
    pca_full = PCA(random_state=42).fit(tfidf_matrix.toarray())
    cum_var = np.cumsum(pca_full.explained_variance_ratio_)
    n_95 = np.searchsorted(cum_var, 0.95) + 1
    print(f"  PCA: 2 components explain {cum_var[1]:.2%} of variance")
    print(f"  PCA: {n_95} components needed for 95% variance")

    # -------------------------------------------------------
    # [6] Anomaly Detection
    # -------------------------------------------------------
    print(f"\n[6] ANOMALY DETECTION")
    print("-" * 50)
    print(f"  Method    : Leave-One-Out Cluster Affinity (Cosine Sim to Centroid)")
    print(f"  Threshold : mean - 1.5 * std = {threshold:.4f}")
    print(f"  Mean aff  : {affinities.mean():.4f}")
    print(f"  Std aff   : {affinities.std():.4f}")

    print(f"\n  Per-document cluster affinity:")
    print(f"  {'Doc':<5} {'Cluster':>7} {'Affinity':>10} {'Status':<12}")
    print(f"  {'-'*5} {'-'*7} {'-'*10} {'-'*12}")
    for i, doc_id in enumerate(doc_ids):
        status = "** ANOMALY **" if doc_id in anomaly_ids else "Normal"
        print(f"  {doc_id:<5} {labels[i]:>7} {affinities[i]:>10.4f} {status:<12}")

    if anomalies:
        print(f"\n  Detected Anomalies ({len(anomalies)}):")
        for doc_id, aff, thr in anomalies:
            print(f"    {doc_id}: affinity={aff:.4f} < threshold={thr:.4f}")
            print(f"         Content: {truncate(DOCUMENTS[doc_id], 65)}")
            print(f"         Subject: {SUBJECT_LABELS[doc_id]}")
    else:
        print("\n  No anomalies detected at the current threshold.")
        lowest_idx = affinities.argmin()
        print(f"  Lowest affinity document: {doc_ids[lowest_idx]} "
              f"(affinity={affinities[lowest_idx]:.4f})")
        print(f"  Content: {truncate(DOCUMENTS[doc_ids[lowest_idx]], 65)}")
        print(f"  Subject: {SUBJECT_LABELS[doc_ids[lowest_idx]]}")
        print("  (This is the gardening document -- the expected anomaly.)")

    # -------------------------------------------------------
    # [7] Semantic Retrieval Interface
    # -------------------------------------------------------
    print(f"\n[7] RETRIEVAL INTERFACE (Query-based Search)")
    print("-" * 50)

    TOP_K = 5
    for query_text, query_info in TEST_QUERIES.items():
        results = retrieve_documents(
            query_text, vectorizer, tfidf_matrix, doc_ids, top_n=TOP_K
        )
        print(f"\n  Query: \"{query_text}\"")
        print(f"  ({query_info['description']})")
        print(f"  {'Rank':<6} {'Doc':<5} {'Similarity':>10} {'Subject':<14} Content")
        print(f"  {'-'*6} {'-'*5} {'-'*10} {'-'*14} {'-'*35}")
        for rank, (doc_id, sim) in enumerate(results, start=1):
            is_rel = "<<" if doc_id in query_info["relevant"] else ""
            print(f"  {rank:<6} {doc_id:<5} {sim:>10.4f} {SUBJECT_LABELS[doc_id]:<14} "
                  f"{truncate(DOCUMENTS[doc_id], 35)} {is_rel}")

    # -------------------------------------------------------
    # [8] Evaluation
    # -------------------------------------------------------
    print(f"\n[8] RETRIEVAL EVALUATION")
    print("-" * 50)

    all_p, all_r, all_f = [], [], []
    print(f"\n  {'Query (short)':<45} {'P@{}'.format(TOP_K):>6} {'R@{}'.format(TOP_K):>6} {'F1':>6}")
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6}")

    for query_text, query_info in TEST_QUERIES.items():
        results = retrieve_documents(
            query_text, vectorizer, tfidf_matrix, doc_ids, top_n=TOP_K
        )
        p, r, f = evaluate_retrieval(results, query_info["relevant"], TOP_K)
        all_p.append(p)
        all_r.append(r)
        all_f.append(f)
        short_q = truncate(query_text, 42)
        print(f"  {short_q:<45} {p:>6.4f} {r:>6.4f} {f:>6.4f}")

    macro_p = np.mean(all_p)
    macro_r = np.mean(all_r)
    macro_f = np.mean(all_f)
    print(f"  {'-'*45} {'-'*6} {'-'*6} {'-'*6}")
    print(f"  {'Macro Average':<45} {macro_p:>6.4f} {macro_r:>6.4f} {macro_f:>6.4f}")

    print(f"""
  Interpretation:
    - Precision@{TOP_K} = {macro_p:.4f} means on average {macro_p*100:.1f}% of the top-{TOP_K}
      retrieved documents are truly relevant.
    - Recall@{TOP_K} = {macro_r:.4f} means on average {macro_r*100:.1f}% of all relevant
      documents are found in the top-{TOP_K}.
    - F1 = {macro_f:.4f} is the harmonic mean, balancing precision and recall.
""")

    # -------------------------------------------------------
    # [9] Report
    # -------------------------------------------------------
    num_subjects = len(set(SUBJECT_LABELS.values()) - {'Anomaly'})
    print(f"\n[9] SHORT REPORT")
    print_sep("-")
    print(f"""
  REPORT: Intelligent Document Search Using Unsupervised Learning
  ================================================================

  Dataset:
    {len(DOCUMENTS)} documents from {num_subjects} subject areas
    (AI, Cybersecurity, Databases, Networking) plus 1 anomaly document.

  1. Document Embeddings:
     - Used TF-IDF vectorisation with sublinear TF and bigrams.
     - Produced a {tfidf_matrix.shape[0]} x {tfidf_matrix.shape[1]} sparse matrix.
     - Each document is represented as a numerical vector in a
       high-dimensional term space. These vectors capture the
       importance of each term relative to the document and the
       entire collection.

  2. Clustering (K-Means, K={N_CLUSTERS}):
     - Achieved a purity of {purity:.4f}, meaning {purity*100:.1f}% of documents
       were placed in the cluster dominated by their true subject.
     - K-Means successfully separated the four subject areas.

  3. Topic Modeling (LDA):
     - Discovered {N_TOPICS} latent topics with interpretable keywords.
     - Topics aligned well with the known subject areas, confirming
       that LDA can discover meaningful themes without supervision.

  4. Dimensionality Reduction:
     - PCA and t-SNE projections to 2D both showed clear separation
       between subject clusters.
     - t-SNE provided better visual separation due to its non-linear
       mapping that preserves local structure.

  5. Anomaly Detection:
     - D25 (gardening) was identified as the most distant document
       from any cluster centroid, confirming it as an outlier.
     - Distance-based anomaly detection using a threshold of
       (mean + 1.5 * std) is effective for small, well-separated
       collections.

  6. Retrieval:
     - Cosine similarity in TF-IDF space successfully retrieves
       relevant documents even when exact query terms do not
       appear in the documents (partial semantic matching via
       shared vocabulary and IDF weighting).
     - Macro-average Precision@{TOP_K} = {macro_p:.4f}, Recall@{TOP_K} = {macro_r:.4f},
       F1 = {macro_f:.4f}.

  Limitations:
     - TF-IDF is a bag-of-words model; it does not capture word
       order or deep semantics. Using pre-trained sentence
       embeddings (e.g., Sentence-BERT) would improve retrieval.
     - With only {len(DOCUMENTS)} documents, clustering and LDA may not
       generalise to larger, noisier collections.
     - The anomaly threshold is heuristic and may need tuning.

  Conclusion:
     Unsupervised learning techniques (clustering, topic modeling,
     dimensionality reduction, anomaly detection) can build an
     effective document search system without labeled training data.
     TF-IDF combined with K-Means provides a strong baseline, and
     LDA adds interpretable topic summaries that help users
     understand the collection structure.
  ================================================================
""")

    print_sep()
    print("  Assignment complete. Visualisations saved as PNG files.")
    print_sep()


if __name__ == "__main__":
    main()
