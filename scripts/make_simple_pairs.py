import requests
import pandas as pd
from pathlib import Path
BASE_URL = "https://raw.githubusercontent.com/koc-lab/lex-simple/main"

FILES = {
    "original": "original_sentences.txt",
    "ref1": "reference_file_1.txt",
    "ref2": "reference_file_2.txt",
    "ref3": "reference_file_3.txt",
}

RAW_DIR = Path("data/simple_raw")
RAW_DIR.mkdir(parents=True, exist_ok=True)

def download_file(filename):
    url = f"{BASE_URL}/{filename}"
    out_path = RAW_DIR / filename
    print(f"Downloading {url}")
    r = requests.get(url, timeout=30)
    r.raise_for_status()
    out_path.write_text(r.text, encoding="utf-8")
    return out_path

def main():
    # download original sentences + 3 human simplification files
    orig_path = download_file(FILES["original"])
    ref_paths = [download_file(FILES[k]) for k in ("ref1", "ref2", "ref3")]

    orig_lines = orig_path.read_text(encoding="utf-8").splitlines()

    rows = []
    pair_id = 1

    # each reference file is another human simplification of the same sentence
    for ref_name, ref_path in zip(("ref1", "ref2", "ref3"), ref_paths):
        ref_lines = ref_path.read_text(encoding="utf-8").splitlines()
        n = min(len(orig_lines), len(ref_lines))

        for i in range(n):
            src = orig_lines[i].strip()
            tgt = ref_lines[i].strip()
            if not src or not tgt:
                continue

            rows.append({
                "id": pair_id,
                "src_legal": src,
                "tgt_plain": tgt,
            })
            pair_id += 1

    df = pd.DataFrame(rows, columns=["id", "src_legal", "tgt_plain"])
    out_path = Path("data/simple_pairs.csv")
    out_path.parent.mkdir(exist_ok=True)
    df.to_csv(out_path, index=False)
    print("Wrote", out_path, "with", len(df), "pairs")

if __name__ == "__main__":
    main()