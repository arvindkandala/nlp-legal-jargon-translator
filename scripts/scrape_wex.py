import re
from tqdm import tqdm
from scripts.common import get, soupify, write_jsonl, now_iso, text_or_none, JSONL_DIR

BASE = "https://www.law.cornell.edu"

def letter_pages():
    # /wex/all/a.html ... /wex/all/z.html
    return [f"{BASE}/wex/all/{ch}.html" for ch in "abcdefghijklmnopqrstuvwxyz"]

def collect_links(letter_url):
    html = get(letter_url)
    s = soupify(html)
    links = []
    for a in s.select("a[href]"):
        href = a.get("href", "")
        # entry pages look like /wex/affidavit, /wex/abandonment, etc.
        if href.startswith("/wex/") and "all/" not in href:
            links.append(BASE + href)
    return sorted(set(links))

def extract_entry(url):
    html = get(url)
    s = soupify(html)
    term = text_or_none(s.select_one("h1"))
    main = s.find("main") or s
    p = main.find("p")
    definition = text_or_none(p)
    return {
        "source": "law.cornell.edu/wex",
        "url": url,
        "term": term,
        "definition_text": definition,
        "scraped_at": now_iso()
    }

def main():
    items = []
    for lu in tqdm(letter_pages(), desc="letters"):
        try:
            links = collect_links(lu)
        except Exception as e:
            print("skip letter", lu, e)
            continue

        for url in tqdm(links, leave=False, desc="terms"):
            try:
                it = extract_entry(url)
                if it["term"] and it["definition_text"]:
                    items.append(it)
            except Exception as e:
                print("skip", url, e)

    out = JSONL_DIR / "wex.jsonl"
    write_jsonl(out, items)
    print("wrote", out, len(items))

if __name__ == "__main__":
    main()
