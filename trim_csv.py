import pandas as pd
from pathlib import Path

INPUT_CSV  = Path("data/combined_pairs.csv")
OUTPUT_CSV = Path("data/combined_pairs_no_tb.csv")  # path for trimmed CSV
N_ROWS     = 1800  # number of data rows to keep (not counting header)

def main():
    df = pd.read_csv(INPUT_CSV)
    df_trimmed = df.iloc[:N_ROWS].copy()
    df_trimmed.to_csv(OUTPUT_CSV, index=False)
    print(f"Saved first {N_ROWS} rows to {OUTPUT_CSV}")

if __name__ == "__main__":
    main()
