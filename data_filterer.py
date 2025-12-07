import pandas as pd
from difflib import SequenceMatcher
import numpy as np

# Load data
INPUT_FILE = 'data/combined_training_data.csv'
OUTPUT_FILE = 'data/combined_training_data.csv'

print(f"Loading {INPUT_FILE}...")
df = pd.read_csv(INPUT_FILE)
print(f"Original size: {len(df)} rows")

# 1. Remove rows where source and target are identical or empty
df = df.dropna(subset=['src_legal', 'tgt_plain'])
df['src_legal'] = df['src_legal'].astype(str).str.strip()
df['tgt_plain'] = df['tgt_plain'].astype(str).str.strip()

# 2. Calculate similarity ratio
def similarity_ratio(text1, text2):
    return SequenceMatcher(None, text1.lower(), text2.lower()).ratio()

print("Calculating similarity scores...")
df['similarity'] = df.apply(lambda row: similarity_ratio(row['src_legal'], row['tgt_plain']), axis=1)

# 3. Filter criteria:
# - Similarity must be < 0.9 (keep sentences that are at least 10% different)
# - BUT keep the first 600 rows (your real data) regardless of similarity to preserve validation set
REAL_DATA_COUNT = 600

real_data = df.iloc[:REAL_DATA_COUNT].copy()
synthetic_data = df.iloc[REAL_DATA_COUNT:].copy()

print(f"\nAnalyzing Synthetic Data ({len(synthetic_data)} rows):")
print(f"  Average similarity: {synthetic_data['similarity'].mean():.4f}")
print(f"  >90% similar: {len(synthetic_data[synthetic_data['similarity'] > 0.9])} rows")
print(f"  >95% similar: {len(synthetic_data[synthetic_data['similarity'] > 0.95])} rows")

# Filter synthetic data
# Strategy: Remove only the most egregious copies (>90% similar)
# We don't want to be too aggressive (e.g. <0.8) or we might lose too much data
synthetic_filtered = synthetic_data[synthetic_data['similarity'] < 0.9].copy()

print(f"\nFiltering Synthetic Data:")
print(f"  Removed {len(synthetic_data) - len(synthetic_filtered)} near-identical rows")
print(f"  Remaining synthetic rows: {len(synthetic_filtered)}")

# 4. Recombine
# IMPORTANT: Reset index to ensure clean concatenation
df_final = pd.concat([real_data, synthetic_filtered], ignore_index=True)

# 5. Save
df_final[['src_legal', 'tgt_plain']].to_csv(OUTPUT_FILE, index=False)
print(f"\n✓ Saved clean dataset to {OUTPUT_FILE}")
print(f"  Total rows: {len(df_final)} ({len(real_data)} real + {len(synthetic_filtered)} synthetic)")
