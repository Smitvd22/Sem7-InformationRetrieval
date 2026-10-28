"""
Practical Assignment 4: Search Engine Ranking and Evaluation
==============================================================
Application : Mini Search Engine for a College Library
Student ID  : U23AI118
Course      : Information Retrieval

Concepts Covered:
  - Vector Space Model (VSM)
  - TF-IDF Weighting
  - Probabilistic Retrieval Model (BM25)
  - Binary Independence Model (BIM)
  - Probabilistic Ranking Principle (PRP)
  - Evaluation Metrics: Precision, Recall, F-measure, MRR, MAP, NDCG
"""

import math
import re
from collections import Counter

# ============================================================
# 1. DOCUMENT COLLECTION  (12 college library documents)
# ============================================================

DOCUMENTS = {
    "D1":  "Machine learning techniques in medical diagnosis and clinical decision support systems",
    "D2":  "Healthcare data analysis using artificial intelligence and big data analytics",
    "D3":  "History of computer networks and the evolution of the internet",
    "D4":  "Deep learning models for disease prediction and early diagnosis in healthcare",
    "D5":  "Database management systems and relational algebra fundamentals",
    "D6":  "Natural language processing applications in electronic health records",
    "D7":  "Introduction to algorithms and computational complexity theory",
    "D8":  "AI-powered drug discovery and personalized medicine using machine learning",
    "D9":  "Cloud computing infrastructure and virtualization technologies",
    "D10": "Reinforcement learning applications in robotic surgery and rehabilitation",
    "D11": "Cyber security threats and intrusion detection systems",
    "D12": "Predictive analytics and machine learning in patient outcome forecasting",
}

# Ground-truth relevance judgments for the query "machine learning in healthcare"
# Graded relevance: 0 = not relevant, 1 = marginally relevant, 2 = relevant, 3 = highly relevant
RELEVANCE_JUDGMENTS = {
    "D1":  3,   # Highly relevant – ML + medical diagnosis
    "D2":  2,   # Relevant – AI + healthcare data
    "D3":  0,   # Not relevant
    "D4":  3,   # Highly relevant – deep learning + disease prediction + healthcare
    "D5":  0,   # Not relevant
    "D6":  2,   # Relevant – NLP + health records
    "D7":  0,   # Not relevant
    "D8":  3,   # Highly relevant – ML + drug discovery + medicine
    "D9":  0,   # Not relevant
    "D10": 2,   # Relevant – RL + robotic surgery
    "D11": 0,   # Not relevant
    "D12": 3,   # Highly relevant – ML + patient outcome
}

# Binary relevance (threshold: graded relevance >= 2 is "relevant")
BINARY_RELEVANCE = {doc_id: (1 if rel >= 2 else 0) for doc_id, rel in RELEVANCE_JUDGMENTS.items()}

QUERY = "machine learning applications in healthcare"

# ============================================================
# 2. TEXT PREPROCESSING
# ============================================================

STOP_WORDS = {
    "a", "an", "the", "in", "of", "on", "and", "or", "to", "for",
    "is", "it", "at", "by", "from", "with", "as", "this", "that",
    "are", "was", "were", "be", "been", "being", "have", "has", "had",
    "do", "does", "did", "will", "would", "shall", "should", "may",
    "might", "can", "could", "not", "but", "if", "than", "so", "no",
    "up", "about", "into", "through", "during", "before", "after",
    "above", "below", "between", "under", "over", "such", "each",
    "its", "using",
}

def tokenize(text):
    """Lowercase, extract alphabetic words, and remove stop words."""
    tokens = re.findall(r'\b[a-z]+\b', text.lower())
    return [t for t in tokens if t not in STOP_WORDS]

def build_vocabulary(documents, query_tokens):
    """Build a sorted vocabulary from all documents and the query."""
    vocab = set()
    for text in documents.values():
        vocab.update(tokenize(text))
    vocab.update(query_tokens)
    return sorted(vocab)

# ============================================================
# 3. RANKING MODELS
# ============================================================

# ---------- 3a. TF-IDF + Vector Space Model (VSM) ----------

def compute_tf(tokens):
    """Compute term frequency (TF) for a list of tokens. Uses augmented TF."""
    tf_counts = Counter(tokens)
    max_freq = max(tf_counts.values()) if tf_counts else 1
    return {term: count / max_freq for term, count in tf_counts.items()}

def compute_idf(documents, vocabulary):
    """Compute inverse document frequency (IDF) for each term in the vocabulary."""
    N = len(documents)
    idf = {}
    for term in vocabulary:
        df = sum(1 for text in documents.values() if term in tokenize(text))
        idf[term] = math.log10((N + 1) / (df + 1)) + 1   # Smoothed IDF
    return idf

def tfidf_vector(tokens, idf, vocabulary):
    """Construct a TF-IDF weighted vector for a token list."""
    tf = compute_tf(tokens)
    return [tf.get(term, 0) * idf.get(term, 0) for term in vocabulary]

def cosine_similarity(vec_a, vec_b):
    """Compute cosine similarity between two equal-length vectors."""
    dot = sum(a * b for a, b in zip(vec_a, vec_b))
    mag_a = math.sqrt(sum(a * a for a in vec_a))
    mag_b = math.sqrt(sum(b * b for b in vec_b))
    if mag_a == 0 or mag_b == 0:
        return 0.0
    return dot / (mag_a * mag_b)

def rank_vsm_tfidf(documents, query_tokens, vocabulary, idf):
    """Rank documents using VSM with TF-IDF weighting and cosine similarity."""
    query_vec = tfidf_vector(query_tokens, idf, vocabulary)
    scores = {}
    for doc_id, text in documents.items():
        doc_vec = tfidf_vector(tokenize(text), idf, vocabulary)
        scores[doc_id] = cosine_similarity(query_vec, doc_vec)
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ---------- 3b. Probabilistic Retrieval Model (BM25) ----------

def rank_bm25(documents, query_tokens, k1=1.5, b=0.75):
    """
    Rank documents using Okapi BM25 (a prominent probabilistic retrieval model).
    BM25 estimates the probability that a document is relevant to the query.
    """
    N = len(documents)
    doc_tokens = {doc_id: tokenize(text) for doc_id, text in documents.items()}
    doc_lengths = {doc_id: len(tokens) for doc_id, tokens in doc_tokens.items()}
    avgdl = sum(doc_lengths.values()) / N

    # Document frequency for each query term
    df = {}
    for term in query_tokens:
        df[term] = sum(1 for tokens in doc_tokens.values() if term in tokens)

    scores = {}
    for doc_id, tokens in doc_tokens.items():
        score = 0.0
        tf_counts = Counter(tokens)
        for term in query_tokens:
            if term not in tf_counts:
                continue
            tf_val = tf_counts[term]
            idf_val = math.log((N - df[term] + 0.5) / (df[term] + 0.5) + 1)
            numerator = tf_val * (k1 + 1)
            denominator = tf_val + k1 * (1 - b + b * doc_lengths[doc_id] / avgdl)
            score += idf_val * (numerator / denominator)
        scores[doc_id] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ---------- 3c. Binary Independence Model (BIM) ----------

def rank_bim(documents, query_tokens, binary_relevance):
    """
    Rank documents using the Binary Independence Model.
    Uses binary term presence/absence and relevance judgments to estimate
    retrieval status values (RSVs) based on log-odds of relevance.
    """
    N = len(documents)
    doc_tokens = {doc_id: set(tokenize(text)) for doc_id, text in documents.items()}

    # Count relevant documents
    R = sum(binary_relevance.values())

    scores = {}
    for doc_id, terms_set in doc_tokens.items():
        score = 0.0
        for term in query_tokens:
            # s = number of relevant docs containing term
            s = sum(1 for d_id, tokens in doc_tokens.items()
                    if term in tokens and binary_relevance.get(d_id, 0) == 1)
            # n = total docs containing term
            n = sum(1 for tokens in doc_tokens.values() if term in tokens)

            if term in terms_set:
                # Log-odds weight with Laplace smoothing to avoid zero denominators
                p_term_rel = (s + 0.5) / (R + 1)
                p_term_nrel = (n - s + 0.5) / (N - R + 1)
                if p_term_rel > 0 and p_term_nrel > 0:
                    weight = math.log(p_term_rel / p_term_nrel)
                else:
                    weight = 0.0
                score += weight
        scores[doc_id] = score
    return sorted(scores.items(), key=lambda x: x[1], reverse=True)

# ============================================================
# 4. EVALUATION METRICS
# ============================================================

def precision_at_k(ranked_list, binary_rel, k):
    """Precision@k: fraction of top-k documents that are relevant."""
    top_k = [doc_id for doc_id, _ in ranked_list[:k]]
    relevant_retrieved = sum(binary_rel.get(d, 0) for d in top_k)
    return relevant_retrieved / k if k > 0 else 0.0

def recall_at_k(ranked_list, binary_rel, k):
    """Recall@k: fraction of all relevant documents that are in the top-k."""
    total_relevant = sum(binary_rel.values())
    if total_relevant == 0:
        return 0.0
    top_k = [doc_id for doc_id, _ in ranked_list[:k]]
    relevant_retrieved = sum(binary_rel.get(d, 0) for d in top_k)
    return relevant_retrieved / total_relevant

def f_measure(precision, recall, beta=1.0):
    """F-measure (harmonic mean of precision and recall)."""
    if precision + recall == 0:
        return 0.0
    return (1 + beta**2) * (precision * recall) / (beta**2 * precision + recall)

def mean_reciprocal_rank(ranked_list, binary_rel):
    """MRR: reciprocal of the rank of the first relevant document."""
    for i, (doc_id, _) in enumerate(ranked_list):
        if binary_rel.get(doc_id, 0) == 1:
            return 1.0 / (i + 1)
    return 0.0

def average_precision(ranked_list, binary_rel):
    """AP: average of precision values computed at each relevant document position."""
    num_relevant = 0
    sum_precision = 0.0
    for i, (doc_id, _) in enumerate(ranked_list):
        if binary_rel.get(doc_id, 0) == 1:
            num_relevant += 1
            sum_precision += num_relevant / (i + 1)
    total_relevant = sum(binary_rel.values())
    if total_relevant == 0:
        return 0.0
    return sum_precision / total_relevant

def dcg_at_k(ranked_list, graded_rel, k):
    """DCG@k: Discounted Cumulative Gain at rank k."""
    dcg = 0.0
    for i, (doc_id, _) in enumerate(ranked_list[:k]):
        rel = graded_rel.get(doc_id, 0)
        dcg += (2**rel - 1) / math.log2(i + 2)   # i+2 because log2(1) = 0
    return dcg

def ndcg_at_k(ranked_list, graded_rel, k):
    """NDCG@k: Normalized DCG at rank k."""
    # Ideal ranking: sort documents by graded relevance descending
    ideal_ranking = sorted(graded_rel.items(), key=lambda x: x[1], reverse=True)
    ideal_ranking_formatted = [(doc_id, rel) for doc_id, rel in ideal_ranking]
    idcg = dcg_at_k(ideal_ranking_formatted, graded_rel, k)
    if idcg == 0:
        return 0.0
    return dcg_at_k(ranked_list, graded_rel, k) / idcg

def evaluate_model(model_name, ranked_list, binary_rel, graded_rel, k=5):
    """Compute all evaluation metrics for a single model and return as a dict."""
    p = precision_at_k(ranked_list, binary_rel, k)
    r = recall_at_k(ranked_list, binary_rel, k)
    f1 = f_measure(p, r)
    mrr = mean_reciprocal_rank(ranked_list, binary_rel)
    ap = average_precision(ranked_list, binary_rel)
    ndcg = ndcg_at_k(ranked_list, graded_rel, k)

    return {
        "Model": model_name,
        f"Precision@{k}": p,
        f"Recall@{k}": r,
        "F1-measure": f1,
        "MRR": mrr,
        "AP (MAP)": ap,
        f"NDCG@{k}": ndcg,
    }

# ============================================================
# 5. DISPLAY HELPERS
# ============================================================

def print_separator(char="=", width=80):
    print(char * width)

def print_ranked_results(model_name, ranked_list, documents, graded_rel, top_n=5):
    """Pretty-print the top-N ranked documents for a given model."""
    print(f"\n  Top {top_n} Results for '{model_name}':")
    print(f"  {'Rank':<6}{'Doc ID':<8}{'Score':<12}{'Rel':>4}  Content")
    print(f"  {'-'*6}{'-'*8}{'-'*12}{'-'*4}  {'-'*40}")
    for rank, (doc_id, score) in enumerate(ranked_list[:top_n], start=1):
        rel_label = graded_rel.get(doc_id, 0)
        content = documents[doc_id][:55] + "..." if len(documents[doc_id]) > 55 else documents[doc_id]
        print(f"  {rank:<6}{doc_id:<8}{score:<12.4f}{rel_label:>4}  {content}")

def print_metrics_table(metrics_list, k=5):
    """Print a comparison table of evaluation metrics across models."""
    headers = ["Model", f"P@{k}", f"R@{k}", "F1", "MRR", "AP", f"NDCG@{k}"]
    col_widths = [25, 8, 8, 8, 8, 8, 10]

    # Header row
    header_row = "  ".join(h.center(w) for h, w in zip(headers, col_widths))
    print(f"  {header_row}")
    print(f"  {'-' * (sum(col_widths) + (len(headers) - 1) * 2)}")

    # Data rows
    for m in metrics_list:
        values = [
            m["Model"],
            f"{m[f'Precision@{k}']:.4f}",
            f"{m[f'Recall@{k}']:.4f}",
            f"{m['F1-measure']:.4f}",
            f"{m['MRR']:.4f}",
            f"{m['AP (MAP)']:.4f}",
            f"{m[f'NDCG@{k}']:.4f}",
        ]
        row = "  ".join(str(v).center(w) for v, w in zip(values, col_widths))
        print(f"  {row}")

# ============================================================
# 6. MAIN EXECUTION
# ============================================================

def main():
    print_separator()
    print("  Practical Assignment 4: Search Engine Ranking and Evaluation")
    print("  Application: Mini Search Engine for a College Library")
    print_separator()

    # --- Document Collection ---
    print("\n[1] DOCUMENT COLLECTION")
    print("-" * 40)
    for doc_id, content in DOCUMENTS.items():
        rel_flag = "Yes" if BINARY_RELEVANCE[doc_id] == 1 else "No"
        print(f"  {doc_id}: {content}")
        print(f"       Relevant to query? {rel_flag}  (Graded: {RELEVANCE_JUDGMENTS[doc_id]})")
    print(f"\n  Total documents: {len(DOCUMENTS)}")
    print(f"  Relevant documents: {sum(BINARY_RELEVANCE.values())}")
    print(f"  Non-relevant documents: {len(DOCUMENTS) - sum(BINARY_RELEVANCE.values())}")

    # --- Query ---
    print(f"\n[2] QUERY")
    print("-" * 40)
    query_tokens = tokenize(QUERY)
    print(f"  Raw query    : \"{QUERY}\"")
    print(f"  Query tokens : {query_tokens}")

    # --- Build Vocabulary & IDF ---
    vocabulary = build_vocabulary(DOCUMENTS, query_tokens)
    idf = compute_idf(DOCUMENTS, vocabulary)

    # --- Ranking with each model ---
    print(f"\n[3] RANKING MODELS")
    print_separator("-")

    # 3a. VSM + TF-IDF
    ranked_vsm = rank_vsm_tfidf(DOCUMENTS, query_tokens, vocabulary, idf)
    print_ranked_results("Vector Space Model (TF-IDF + Cosine)", ranked_vsm, DOCUMENTS, RELEVANCE_JUDGMENTS)

    # 3b. BM25
    ranked_bm25 = rank_bm25(DOCUMENTS, query_tokens)
    print_ranked_results("Probabilistic Model (BM25)", ranked_bm25, DOCUMENTS, RELEVANCE_JUDGMENTS)

    # 3c. BIM
    ranked_bim = rank_bim(DOCUMENTS, query_tokens, BINARY_RELEVANCE)
    print_ranked_results("Binary Independence Model (BIM)", ranked_bim, DOCUMENTS, RELEVANCE_JUDGMENTS)

    # --- Probabilistic Ranking Principle ---
    print(f"\n[4] PROBABILISTIC RANKING PRINCIPLE (PRP)")
    print_separator("-")
    print("""
  The Probabilistic Ranking Principle (PRP) states that:

    "For optimal retrieval effectiveness, documents should be ranked in
     decreasing order of their probability of relevance to the query."

  Key implications for our ranking models:

  1. BM25 (Probabilistic Retrieval Model):
     - BM25 directly operationalises the PRP by estimating the probability
       that each document is relevant, based on term frequencies, document
       length normalisation, and inverse document frequency.
     - Documents with higher estimated relevance probability are ranked first.

  2. Binary Independence Model (BIM):
     - BIM is the simplest probabilistic model. It assumes terms are
       independent given relevance. It uses binary presence/absence of terms
       and relevance feedback to compute log-odds retrieval status values.
     - Ranking by RSV is equivalent to ranking by P(relevant | document).

  3. Vector Space Model (TF-IDF):
     - VSM does not directly estimate relevance probabilities; instead it
       uses geometric similarity (cosine) in a term-weight space.
     - However, under certain assumptions about term distributions, TF-IDF
       cosine similarity can approximate a relevance ranking.

  In practice, PRP guides us to prefer models that estimate relevance
  probability more accurately. BM25 typically outperforms simple VSM
  because it models document length and term saturation effects.
""")

    # --- Evaluation ---
    print(f"\n[5] EVALUATION METRICS")
    print_separator("-")

    K = 5  # Evaluate at top-5

    metrics_vsm  = evaluate_model("VSM (TF-IDF + Cosine)", ranked_vsm, BINARY_RELEVANCE, RELEVANCE_JUDGMENTS, K)
    metrics_bm25 = evaluate_model("Probabilistic (BM25)",  ranked_bm25, BINARY_RELEVANCE, RELEVANCE_JUDGMENTS, K)
    metrics_bim  = evaluate_model("Binary Independence (BIM)", ranked_bim, BINARY_RELEVANCE, RELEVANCE_JUDGMENTS, K)

    all_metrics = [metrics_vsm, metrics_bm25, metrics_bim]

    print(f"\n  Evaluation at K = {K}")
    print()
    print_metrics_table(all_metrics, K)

    # Detailed per-model breakdown
    for m in all_metrics:
        print(f"\n  --- {m['Model']} ---")
        print(f"    Precision@{K} = {m[f'Precision@{K}']:.4f}  "
              f"({int(m[f'Precision@{K}'] * K)} of top-{K} are relevant)")
        print(f"    Recall@{K}    = {m[f'Recall@{K}']:.4f}  "
              f"({int(m[f'Recall@{K}'] * sum(BINARY_RELEVANCE.values()))} of "
              f"{sum(BINARY_RELEVANCE.values())} relevant docs found)")
        print(f"    F1-measure   = {m['F1-measure']:.4f}")
        print(f"    MRR          = {m['MRR']:.4f}  "
              f"(first relevant doc at rank {int(1/m['MRR']) if m['MRR'] > 0 else 'N/A'})")
        print(f"    AP (MAP)     = {m['AP (MAP)']:.4f}")
        print(f"    NDCG@{K}     = {m[f'NDCG@{K}']:.4f}")

    # --- Model Comparison ---
    print(f"\n[6] MODEL COMPARISON AND ANALYSIS")
    print_separator("-")

    # Find the best model by each metric
    metric_keys = [f"Precision@{K}", f"Recall@{K}", "F1-measure", "MRR", "AP (MAP)", f"NDCG@{K}"]
    print("\n  Best Model per Metric:")
    print(f"  {'Metric':<15} {'Best Model':<30} {'Score':<10}")
    print(f"  {'-'*15} {'-'*30} {'-'*10}")
    for mk in metric_keys:
        best = max(all_metrics, key=lambda m: m[mk])
        print(f"  {mk:<15} {best['Model']:<30} {best[mk]:<10.4f}")

    # Overall best (average of normalised metrics)
    for m in all_metrics:
        m["avg_score"] = sum(m[mk] for mk in metric_keys) / len(metric_keys)
    overall_best = max(all_metrics, key=lambda m: m["avg_score"])

    print(f"\n  Overall Best Model (average across all metrics):")
    print(f"    >>> {overall_best['Model']}  (avg score: {overall_best['avg_score']:.4f}) <<<")

    # --- Report ---
    print(f"\n[7] SHORT REPORT")
    print_separator("-")
    print(f"""
  REPORT: Comparison of Ranking Models for College Library Search
  ================================================================

  Objective:
    Evaluate three ranking models — VSM (TF-IDF), BM25, and BIM — on the
    query "machine learning applications in healthcare" over a collection
    of {len(DOCUMENTS)} library documents ({sum(BINARY_RELEVANCE.values())} relevant, {len(DOCUMENTS) - sum(BINARY_RELEVANCE.values())} non-relevant).

  Key Findings:

    1. Vector Space Model (TF-IDF + Cosine Similarity):
       - Represents both query and documents as TF-IDF weighted vectors.
       - Uses cosine similarity to measure relevance.
       - Performs well when query terms appear frequently in relevant docs.
       - Limitation: Does not account for document length normalisation
         as effectively as BM25.

    2. Probabilistic Retrieval Model (BM25):
       - Estimates relevance probability using term frequency saturation
         and document length normalisation.
       - Generally produces the best rankings because it balances term
         importance with document characteristics.
       - Directly follows the Probabilistic Ranking Principle.

    3. Binary Independence Model (BIM):
       - Uses only binary term presence/absence.
       - Requires relevance feedback (known relevant documents) to
         compute effective weights.
       - Simpler but loses fine-grained term frequency information.

  Conclusion:
    The {overall_best['Model']} model achieved the highest overall
    performance with an average metric score of {overall_best['avg_score']:.4f}.

    BM25 is generally recommended for practical search engine deployment
    because it:
      (a) follows the Probabilistic Ranking Principle,
      (b) handles varying document lengths gracefully, and
      (c) models term frequency saturation (diminishing returns of
          repeated term occurrences).

    VSM with TF-IDF remains a strong baseline, while BIM is most useful
    when explicit relevance feedback is available from users.
  ================================================================
""")

    print_separator()
    print("  Assignment complete. All ranking models executed and evaluated.")
    print_separator()


if __name__ == "__main__":
    main()
