import pandas as pd

# Load data
df = pd.read_csv('data/real_pairs.csv')
print(f"Original dataset: {len(df)} rows")

# Check for missing values
print(f"Rows with missing values: {df[['src_legal', 'tgt_plain']].isna().any(axis=1).sum()}")

# Find exact duplicates (where source == target)
df['is_copy'] = df['src_legal'].str.strip() == df['tgt_plain'].str.strip()
copy_pairs = df[df['is_copy']]

print(f"\n{'='*60}")
print(f"Found {len(copy_pairs)} rows where Source == Target")
print(f"{'='*60}")

# Show examples of what will be removed
if len(copy_pairs) > 0:
    print("\nExamples of copy-paste pairs (will be removed):")
    for i, row in copy_pairs.head(5).iterrows():
        print(f"\nRow {row['id'] if 'id' in row else i}:")
        print(f"  Legal:  {row['src_legal'][:100]}...")
        print(f"  Plain:  {row['tgt_plain'][:100]}...")

# Remove copy-paste pairs
df_clean = df[~df['is_copy']].drop(columns=['is_copy'])

print(f"\n{'='*60}")
print(f"Cleaned dataset: {len(df_clean)} rows")
print(f"Removed: {len(df) - len(df_clean)} copy-paste pairs")
print(f"{'='*60}")

# Save cleaned version
output_path = 'data/real_pairs_cleaned.csv'
df_clean.to_csv(output_path, index=False)
print(f"\n✓ Saved cleaned data to {output_path}")

# Show statistics
print(f"\nDataset Statistics:")
print(f"  - Unique source sentences: {df_clean['src_legal'].nunique()}")
print(f"  - Unique target sentences: {df_clean['tgt_plain'].nunique()}")
print(f"  - Average source length: {df_clean['src_legal'].str.len().mean():.1f} chars")
print(f"  - Average target length: {df_clean['tgt_plain'].str.len().mean():.1f} chars")
