import re
from config import KEYWORDS, SEVERITY_KEYWORDS
from utils import extract_cves, extract_techniques

CVSS_RE = re.compile(r"(?:CVSS[^\d]{0,10})(\d{1,2}\.\d)", re.IGNORECASE)

def extract_cvss(text):
    m = CVSS_RE.search(text or "")
    if m:
        try:
            return float(m.group(1))
        except ValueError:
            return None
    return None

def weak_threat_type(text):
    t = (text or "").lower()
    scores = {k: sum(t.count(kw) for kw in kws) for k, kws in KEYWORDS.items()}
    best = max(scores, key=scores.get)
    return best if scores[best] > 0 else "other"

def weak_severity(text, cvss=None):
    if cvss is not None:
        if cvss >= 9:
            return "critical"
        if cvss >= 7:
            return "high"
        if cvss >= 4:
            return "medium"
        return "low"
    t = (text or "").lower()
    for sev in ["critical", "high", "medium", "low"]:
        for kw in SEVERITY_KEYWORDS[sev]:
            if kw in t:
                return sev
    return "medium"

def enrich(rec):
    text = ((rec.get("title") or "") + ". " + (rec.get("text") or "")).strip()
    rec["cve_ids"] = ",".join(extract_cves(text))
    rec["attack_techniques"] = ",".join(extract_techniques(text))
    rec["cvss"] = extract_cvss(text)
    rec["threat_type"] = weak_threat_type(text)
    rec["severity"] = weak_severity(text, rec["cvss"])
    return rec