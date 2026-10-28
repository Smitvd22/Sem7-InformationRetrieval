import requests
import feedparser
import pandas as pd
from config import FEEDS
from database import init_db, insert_report
from utils import make_uid, strip_html, clean_text

HEADERS = {"User-Agent": "ThreatScopeAI/1.0 (+educational)"}

def fetch_feed(name, url, timeout=20):
    try:
        r = requests.get(url, headers=HEADERS, timeout=timeout)
        r.raise_for_status()
        parsed = feedparser.parse(r.content)
    except Exception as e:
        print(f"[ingest] failed {name}: {e}")
        return []
    out = []
    for entry in parsed.entries:
        title = clean_text(getattr(entry, "title", ""))
        link = getattr(entry, "link", "")
        summary = strip_html(getattr(entry, "summary", ""))
        content = ""
        if "content" in entry and entry.content:
            content = strip_html(entry.content[0].get("value", ""))
        text = clean_text(" ".join([summary, content])) or title
        published = getattr(entry, "published", "") or getattr(entry, "updated", "")
        if not title or not link:
            continue
        out.append({
            "uid": make_uid(link, title),
            "title": title,
            "url": link,
            "source": name,
            "published": published,
            "text": text,
        })
    return out

def ingest_all(feeds=FEEDS):
    init_db()
    total = 0
    for name, url in feeds:
        items = fetch_feed(name, url)
        for it in items:
            if insert_report(it):
                total += 1
    print(f"[ingest] inserted {total} new reports")
    return total

def load_sample_data(path="data/sample_reports.csv"):
    init_db()
    df = pd.read_csv(path)
    n = 0
    for _, row in df.iterrows():
        rec = row.to_dict()
        rec["uid"] = make_uid(rec.get("url"), rec.get("title"))
        if insert_report(rec):
            n += 1
    print(f"[ingest] sample rows inserted: {n}")
    return n

if __name__ == "__main__":
    ingest_all()