import pandas as pd
from pathlib import Path

def main():
    df = pd.read_csv(Path("data/combined_pairs.csv"))
    df_trimmed = df.iloc[:1800].copy()
    df_trimmed.to_csv(Path("data/combined_pairs_no_tb.csv"), index=False)
    print(f"Saved first {1800} rows to {Path("data/combined_pairs_no_tb.csv")}")

if __name__ == "__main__":
    main()
