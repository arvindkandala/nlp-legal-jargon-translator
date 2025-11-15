import time, random, re, json, hashlib
from pathlib import Path
from datetime import datetime
import requests
from bs4 import BeautifulSoup

UA = {"User-Agent": "Duke-684-LegalJargon/1.0 (contact: your@duke.edu)"}

BASE_DIR = Path(__file__).resolve().parent.parent
RAW_DIR  = BASE_DIR / "data" / "raw"
JSONL_DIR= BASE_DIR / "data" / "jsonl"
RAW_DIR.mkdir(parents=True, exist_ok=True)
JSONL_DIR.mkdir(parents=True, exist_ok=True)

def sleep_polite(min_s=1.5, max_s=2.5):
    time.sleep(random.uniform(min_s, max_s))

def get(url, session=None, allow_redirects=True, save_raw=False):
    s = session or requests.Session()
    r = s.get(url, headers=UA, timeout=20, allow_redirects=allow_redirects)
    r.raise_for_status()
    html = r.text
    if save_raw:
        h = hashlib.sha1(url.encode()).hexdigest()[:12]
        (RAW_DIR / f"{h}.html").write_text(html, encoding="utf-8")
    sleep_polite()
    return html

def soupify(html):
    return BeautifulSoup(html, "lxml")

def write_jsonl(path, items):
    with open(path, "w", encoding="utf-8") as f:
        for it in items:
            f.write(json.dumps(it, ensure_ascii=False) + "\n")

def now_iso():
    return datetime.utcnow().replace(microsecond=0).isoformat()+"Z"

def norm_space(t):
    return re.sub(r"\s+", " ", t or "").strip()

def text_or_none(node):
    return norm_space(node.get_text(" ", strip=True)) if node else None
