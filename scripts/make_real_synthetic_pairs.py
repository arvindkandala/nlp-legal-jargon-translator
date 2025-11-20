from pathlib import Path
import pandas as pd

DATA_DIR = Path("data")
REAL_PATH = DATA_DIR / "simple_pairs.csv"
SYN_PATH = DATA_DIR / "sample_data.txt"
OUT_PATH = DATA_DIR / "real_synthetic_pairs.csv"


def load_real_pairs():
    """Load real (human) pairs from Lex-Simple."""
    df = pd.read_csv(REAL_PATH)
    df = df[["src_legal", "tgt_plain"]].dropna()
    df["source"] = "real_lex_simple"
    return df


def load_synthetic_pairs():

    # Each line is tab-separated; there may be multiple tabs
    # treat everything except the last tabless chunk as the legal phrase
    # and the last chunk as the plain-English translation
    rows = []
    with SYN_PATH.open("r", encoding="utf-8") as f:
        for line in f:
            line = line.strip()
            if not line:
                continue

            parts = line.split("\t")
            if len(parts) < 2:
                continue

            src = " ".join(parts[:-1]).strip()
            tgt = parts[-1].strip()

            if not src or not tgt:
                continue

            rows.append(
                {
                    "src_legal": src,
                    "tgt_plain": tgt,
                    "source": "synthetic_dict",
                }
            )

    df = pd.DataFrame(rows)
    return df


def main():
    if not REAL_PATH.exists():
        raise FileNotFoundError(f"Real data not found at {REAL_PATH}")
    if not SYN_PATH.exists():
        raise FileNotFoundError(f"Synthetic data not found at {SYN_PATH}")

    real_df = load_real_pairs()
    syn_df = load_synthetic_pairs()

    print("Real pairs:", len(real_df))
    print("Synthetic pairs:", len(syn_df))

    combined = pd.concat([real_df, syn_df], ignore_index=True)

    combined.insert(0, "id", range(1, len(combined) + 1))

    OUT_PATH.parent.mkdir(exist_ok=True)
    combined.to_csv(OUT_PATH, index=False)
    print("Wrote", OUT_PATH, "with", len(combined), "rows")


if __name__ == "__main__":
    main()
