import pandas as pd
from datetime import datetime, timezone
from dateutil import parser as dateparser

def recency_days(published):
    try:
        dt = dateparser.parse(published)
        if dt.tzinfo is None:
            dt = dt.replace(tzinfo=timezone.utc)
        return max(0.0, (datetime.now(timezone.utc) - dt).days)
    except Exception:
        return 30.0

def build_features(reports):
    rows = []
    for r in reports:
        text = (r.get("text") or "") + " " + (r.get("title") or "")
        cves = (r.get("cve_ids") or "").split(",") if r.get("cve_ids") else []
        techs = (r.get("attack_techniques") or "").split(",") if r.get("attack_techniques") else []
        cvss = r.get("cvss")
        rows.append({
            "text_len": len(text),
            "cve_count": len([c for c in cves if c]),
            "tech_count": len([t for t in techs if t]),
            "cvss": float(cvss) if cvss is not None else 5.0,
            "recency_days": recency_days(r.get("published")),
            "has_cve": int(bool(r.get("cve_ids"))),
            "has_tech": int(bool(r.get("attack_techniques"))),
        })
    return pd.DataFrame(rows)