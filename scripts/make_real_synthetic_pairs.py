from pathlib import Path
import pandas as pd

BASE_DIR = Path(__file__).resolve().parents[1]

def main():
    real = pd.read_csv(BASE_DIR / "data" / "combined_pairs.csv")
    synthetic = pd.read_csv(BASE_DIR / "data" / "textbook_pairs.csv", on_bad_lines="skip")
    combined = pd.concat([real, synthetic], ignore_index=True)
    combined.to_csv(BASE_DIR / "data" / "combined_pairs.csv", index=False)

if __name__ == "__main__":
    main()
