from features import recency_days

DEFAULT_WEIGHTS = {"sem": 0.55, "severity": 0.20, "recency": 0.15, "source": 0.10}
SEV_SCORE = {"critical": 1.0, "high": 0.75, "medium": 0.45, "low": 0.2}
SOURCE_REP = {
    "CISA": 1.0,
    "Microsoft Security": 0.95,
    "Google Project Zero": 0.95,
    "SANS ISC": 0.9,
    "Krebs on Security": 0.85,
    "The Hacker News": 0.7,
    "BleepingComputer": 0.7,
}

def rank_results(query, candidates, reports_by_id, weights=None):
    if not candidates:
        return []
    weights = weights or DEFAULT_WEIGHTS
    rows = []
    for c in candidates:
        r = reports_by_id.get(c["id"], {})
        sev = SEV_SCORE.get(r.get("severity") or "medium", 0.45)
        days = recency_days(r.get("published"))
        rec = max(0.0, 1.0 - min(days, 365) / 365)
        src = SOURCE_REP.get(r.get("source") or "", 0.5)
        final = (
            weights["sem"] * c["score"]
            + weights["severity"] * sev
            + weights["recency"] * rec
            + weights["source"] * src
        )
        rows.append({**r, "semantic_score": c["score"], "rank_score": final})
    rows.sort(key=lambda x: -x["rank_score"])
    return rows