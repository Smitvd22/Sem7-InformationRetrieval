import re
import html
import hashlib
from bs4 import BeautifulSoup

CVE_RE = re.compile(r"CVE-\d{4}-\d{4,7}", re.IGNORECASE)
TECH_RE = re.compile(r"\bT\d{4}(?:\.\d{3})?\b")

def strip_html(s):
    return BeautifulSoup(s or "", "html.parser").get_text(" ", strip=True)

def clean_text(s):
    s = html.unescape(s or "")
    s = re.sub(r"\s+", " ", s)
    return s.strip()

def extract_cves(text):
    return sorted({m.group(0).upper() for m in CVE_RE.finditer(text or "")})

def extract_techniques(text):
    return sorted({m.group(0) for m in TECH_RE.finditer(text or "")})

def make_uid(url, title):
    base = (url or "") + "|" + (title or "")
    return hashlib.sha1(base.encode("utf-8")).hexdigest()