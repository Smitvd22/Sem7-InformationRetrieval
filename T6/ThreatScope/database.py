import sqlite3
from contextlib import contextmanager
from config import DB_PATH

SCHEMA = """
CREATE TABLE IF NOT EXISTS reports (
    id INTEGER PRIMARY KEY AUTOINCREMENT,
    uid TEXT UNIQUE,
    title TEXT,
    url TEXT,
    source TEXT,
    published TEXT,
    text TEXT,
    cve_ids TEXT,
    attack_techniques TEXT,
    cvss REAL,
    threat_type TEXT,
    severity TEXT,
    confidence REAL,
    cluster_id INTEGER,
    anomaly_score REAL,
    created_at TEXT DEFAULT CURRENT_TIMESTAMP
);
CREATE INDEX IF NOT EXISTS idx_reports_cluster ON reports(cluster_id);
CREATE INDEX IF NOT EXISTS idx_reports_threat ON reports(threat_type);
"""

@contextmanager
def get_conn():
    conn = sqlite3.connect(DB_PATH)
    conn.row_factory = sqlite3.Row
    try:
        yield conn
        conn.commit()
    finally:
        conn.close()

def init_db():
    with get_conn() as conn:
        conn.executescript(SCHEMA)

def insert_report(rec):
    fields = ["uid", "title", "url", "source", "published", "text",
              "cve_ids", "attack_techniques", "cvss", "threat_type",
              "severity", "confidence", "cluster_id", "anomaly_score"]
    values = [rec.get(f) for f in fields]
    with get_conn() as conn:
        cur = conn.execute(
            f"INSERT OR IGNORE INTO reports ({','.join(fields)}) "
            f"VALUES ({','.join(['?'] * len(fields))})",
            values,
        )
        return cur.lastrowid if cur.rowcount > 0 else None

def fetch_all_reports():
    with get_conn() as conn:
        rows = conn.execute("SELECT * FROM reports ORDER BY id").fetchall()
        return [dict(r) for r in rows]

def fetch_report(rid):
    with get_conn() as conn:
        row = conn.execute("SELECT * FROM reports WHERE id=?", (rid,)).fetchone()
        return dict(row) if row else None

def update_many(updates):
    with get_conn() as conn:
        for rid, fields in updates:
            if not fields:
                continue
            keys = list(fields.keys())
            sql = f"UPDATE reports SET {', '.join(k + '=?' for k in keys)} WHERE id=?"
            conn.execute(sql, [fields[k] for k in keys] + [rid])

def count_reports():
    with get_conn() as conn:
        return conn.execute("SELECT COUNT(*) FROM reports").fetchone()[0]