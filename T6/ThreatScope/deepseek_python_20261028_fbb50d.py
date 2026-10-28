import csv
import random
from pathlib import Path
from datetime import datetime, timedelta
from config import DATA_DIR

random.seed(42)

SOURCES = [
    "CISA", "The Hacker News", "BleepingComputer",
    "Krebs on Security", "Microsoft Security",
    "Google Project Zero", "SANS ISC",
]
THREATS = ["ransomware", "phishing", "malware", "vulnerability", "data_breach", "ddos", "supply_chain"]
SECTORS = ["healthcare", "finance", "energy", "government", "retail", "education", "manufacturing"]
ACTORS = ["LockBit", "BlackCat", "Cl0p", "APT29", "FIN7", "Scattered Spider", "unknown actors"]

TEMPLATES = [
    "A new {threat} campaign has been observed targeting {sector} organizations. "
    "Analysts identified CVE-2024-{cve} exploited in the wild with CVSS {cvss}. "
    "The actors used MITRE ATT&CK technique T{tech} to gain initial access and T{tech2} for persistence.",

    "Researchers disclosed a {threat} incident affecting {sector} providers. "
    "The report references CVE-2023-{cve} with a CVSS score of {cvss}. "
    "Mitigations are available and organizations are urged to patch immediately.",

    "A {actor} {threat} operation was linked to a supply chain compromise. "
    "The investigation found CVE-2022-{cve} (CVSS {cvss}) and techniques T{tech} and T{tech2}. "
    "The campaign has been active for several weeks.",

    "Security teams observed unusual {threat} activity in {sector} networks. "
    "The advisory mentions CVE-2024-{cve}, CVSS {cvss}, and ATT&CK technique T{tech}. "
    "No definitive attribution has been made.",

    "A {threat} incident was reported after attackers exploited a zero-day tracked as CVE-2024-{cve}. "
    "The vulnerability has a CVSS score of {cvss}. "
    "The actors used technique T{tech} to move laterally and T{tech2} for exfiltration.",

    "An advisory warns of a {threat} campaign against {sector} organizations. "
    "The report lists CVE-2023-{cve} and CVSS {cvss}. "
    "Technique T{tech} was observed for initial access.",

    "A {threat} attack disrupted {sector} services and led to data exposure. "
    "The incident involved CVE-2021-{cve} (CVSS {cvss}) and technique T{tech}. "
    "The source recommends network segmentation and monitoring.",
]

def rand_cve():
    return str(random.randint(1000, 9999))

def rand_tech():
    return str(random.randint(1000, 1999))

def rand_cvss():
    return round(random.uniform(3.0, 9.9), 1)

def gen_text():
    return random.choice(TEMPLATES).format(
        threat=random.choice(THREATS),
        sector=random.choice(SECTORS),
        actor=random.choice(ACTORS),
        cve=rand_cve(),
        cvss=rand_cvss(),
        tech=rand_tech(),
        tech2=rand_tech(),
    )

def main(n=80):
    DATA_DIR.mkdir(exist_ok=True)
    path = DATA_DIR / "sample_reports.csv"
    now = datetime.utcnow()
    rows = []
    for i in range(n):
        title = f"{random.choice(ACTORS)} {random.choice(THREATS)} activity targets {random.choice(SECTORS)}"
        rows.append({
            "title": title,
            "url": f"https://example.org/report/{i+1}",
            "source": random.choice(SOURCES),
            "published": (now - timedelta(days=random.randint(0, 180))).isoformat(),
            "text": gen_text(),
        })
    with open(path, "w", newline="", encoding="utf-8") as f:
        writer = csv.DictWriter(f, fieldnames=["title", "url", "source", "published", "text"])
        writer.writeheader()
        writer.writerows(rows)
    print(f"Wrote {n} sample reports to {path}")

if __name__ == "__main__":
    main()