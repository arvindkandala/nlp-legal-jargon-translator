from huggingface_hub import hf_hub_download
import pandas as pd
import re
from pathlib import Path

# download master_clauses.csv from the CUAD dataset on Hugging Face
csv_path = hf_hub_download(
    repo_id="theatticusproject/cuad",
    repo_type="dataset",
    filename="CUAD_v1/master_clauses.csv"  # <- REAL file name
)

print("Downloaded CUAD CSV to:", csv_path)

# read the CSV with pandas
raw = pd.read_csv(csv_path)

# pick a few clause types (columns) we care about
# you can change this list later if you want different clauses
CLAUSE_COLUMNS = {
    "Governing Law": "GOVERNING_LAW",
    "Termination For Convenience": "TERMINATION",
    "Anti-Assignment": "ANTI_ASSIGNMENT",
    "Cap On Liability": "LIABILITY_CAP",
    "License Grant": "LICENSE_GRANT",
}

def clean_text(t: str) -> str:
    """Clean up the clause text a little bit."""
    if not isinstance(t, str):
        return ""
    # remove newlines and extra spaces
    t = re.sub(r"\s+", " ", t).strip()
    # strip simple list-like wrappers
    if t.startswith("['") and t.endswith("']"):
        t = t[2:-2]
    return t.strip()

rows = []
id_counter = 1

for _, row in raw.iterrows():
    for col_name, clause_type in CLAUSE_COLUMNS.items():
        if col_name not in raw.columns:
            continue
        text = clean_text(row[col_name])
        if not text or text == "[]":
            continue

        # keep medium-length snippets
        if 200 < len(text) < 800:
            rows.append({
                "id": id_counter,
                "clause_type": clause_type,
                "src_legal": text,
                "tgt_plain": "",   # you will fill this in by hand
                "notes": "",
            })
            id_counter += 1

        if id_counter > 300:  # stop after ~300 examples
            break
    if id_counter > 300:
        break

df = pd.DataFrame(rows)
Path("data").mkdir(exist_ok=True)
out_path = Path("data/contract_pairs_template.csv")
df.to_csv(out_path, index=False)
print("Wrote", out_path, "with", len(df), "rows")
