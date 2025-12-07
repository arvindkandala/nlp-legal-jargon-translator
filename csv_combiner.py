import pandas as pd
from pathlib import Path

# === CONFIGURATION ===
# List your CSV files in order (real data first is recommended)
csv_files = [
    'data/real_pairs.csv',
    'data/synthetic_pairs.csv',
    'data/extra_400_synthetic_pairs.csv', 
    'data/legal_pairs_supplement.csv',
    'data/textbook_pairs.csv'
    # Add more files as needed
]

output_path = 'data/combined_training_data.csv'

dfs = []

for i, file_path in enumerate(csv_files):
    try:
        df = pd.read_csv(file_path)
        print(f"  ✓ Loaded {file_path}: {len(df)} rows")
        dfs.append(df)
    except FileNotFoundError:
        print(f"  ⚠ Skipping {file_path}: File not found")
    except Exception as e:
        print(f"  ✗ Error loading {file_path}: {e}")

if not dfs:
    print("ERROR: No files loaded successfully!")
    exit(1)

combined_df = pd.concat(dfs, ignore_index=True)

print(f"\nTotal rows: {len(combined_df)}")
print(f"Columns: {list(combined_df.columns)}")

# Basic validation
assert 'src_legal' in combined_df.columns, "Missing 'src_legal' column"
assert 'tgt_plain' in combined_df.columns, "Missing 'tgt_plain' column"

# Remove any rows with missing values
initial_len = len(combined_df)
combined_df = combined_df.dropna(subset=['src_legal', 'tgt_plain'])
if len(combined_df) < initial_len:
    print(f"Removed {initial_len - len(combined_df)} rows with missing values")

# Save
combined_df.to_csv(output_path, index=False)
print(f"\n✓ Saved combined data to {output_path}")
print(f"Final dataset: {len(combined_df)} rows")
