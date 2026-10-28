from pathlib import Path

BASE_DIR = Path(__file__).resolve().parent
DATA_DIR = BASE_DIR / "data"
MODEL_DIR = BASE_DIR / "models"
INDEX_DIR = BASE_DIR / "index"
DB_PATH = DATA_DIR / "threatscope.db"

DATA_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)
INDEX_DIR.mkdir(exist_ok=True)

EMBEDDING_MODEL = "all-MiniLM-L6-v2"
BERT_MODEL = "distilbert-base-uncased"

FEEDS = [
    ("CISA", "https://www.cisa.gov/cybersecurity-advisories/all.xml"),
    ("The Hacker News", "https://feeds.feedburner.com/TheHackersNews"),
    ("BleepingComputer", "https://www.bleepingcomputer.com/feed/"),
    ("Krebs on Security", "https://krebsonsecurity.com/feed/"),
    ("Microsoft Security", "https://www.microsoft.com/security/blog/feed/"),
    ("Google Project Zero", "https://googleprojectzero.blogspot.com/feeds/posts/default?alt=rss"),
    ("SANS ISC", "https://isc.sans.edu/rssfeed.xml"),
]

THREAT_TYPES = [
    "ransomware", "phishing", "malware", "vulnerability",
    "data_breach", "ddos", "insider_threat", "supply_chain", "other",
]

SEVERITY_LEVELS = ["low", "medium", "high", "critical"]

KEYWORDS = {
    "ransomware": ["ransomware", "ransom", "lockbit", "conti", "revil", "extortion"],
    "phishing": ["phishing", "spearphishing", "credential theft", "smishing"],
    "malware": ["malware", "trojan", "backdoor", "rootkit", "spyware", "worm"],
    "vulnerability": ["vulnerability", "cve-", "zero-day", "patch", "exploit"],
    "data_breach": ["data breach", "leaked", "exfiltration", "records exposed"],
    "ddos": ["ddos", "denial of service", "botnet flood"],
    "insider_threat": ["insider", "rogue employee"],
    "supply_chain": ["supply chain", "third-party", "dependency"],
}

SEVERITY_KEYWORDS = {
    "critical": ["critical", "cvss 9", "cvss 10", "emergency", "actively exploited", "rce"],
    "high": ["high", "cvss 7", "cvss 8", "severe", "remote code execution"],
    "medium": ["medium", "moderate", "cvss 4", "cvss 5", "cvss 6"],
    "low": ["low", "minor", "informational", "cvss 1", "cvss 2", "cvss 3"],
}