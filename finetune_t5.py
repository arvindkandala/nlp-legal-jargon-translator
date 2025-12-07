import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset, load_dataset, concatenate_datasets
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    T5Config
)
import evaluate
from sklearn.model_selection import train_test_split

# --- CONFIGURATION ---
DATA_PATH = Path("data/real_pairs.csv")
MODEL_DIR = Path("models/t5_legal_simplifier")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "t5-base"
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 15          # Enough for ~4000 examples
LEARNING_RATE = 4e-5     # Slightly lower LR for stability

# --- DATA PREPARATION ---
print("="*60)
print("LOADING & AUGMENTING DATA")
print("="*60)

# 1. Load YOUR Real Pairs
try:
    df_real = pd.read_csv(DATA_PATH)
    # Ensure columns exist
    if 'src_legal' not in df_real.columns or 'tgt_plain' not in df_real.columns:
        raise ValueError("real_pairs.csv must have 'src_legal' and 'tgt_plain' columns")
    
    # Filter out exact duplicates (where source == target)
    df_real = df_real[df_real['src_legal'].str.strip() != df_real['tgt_plain'].str.strip()]
    
    # Convert to Hugging Face Dataset
    ds_real = Dataset.from_pandas(df_real[['src_legal', 'tgt_plain']])
    print(f"✓ Loaded {len(ds_real)} examples from real_pairs.csv")
except Exception as e:
    print(f"⚠ Could not load real_pairs.csv: {e}")
    ds_real = None

# 2. Load Manor & Li (Legal Summarization) - High Quality Legal
print("\nDownloading 'mteb/legal_summarization' (Manor & Li)...")
try:
    ds_legal = load_dataset("mteb/legal_summarization", split="test") # It only has a 'test' split usually
    # Rename columns to match yours
    ds_legal = ds_legal.rename_column("text", "src_legal")
    ds_legal = ds_legal.rename_column("summary", "tgt_plain")
    ds_legal = ds_legal.remove_columns([c for c in ds_legal.column_names if c not in ['src_legal', 'tgt_plain']])
    print(f"✓ Added {len(ds_legal)} examples from Manor & Li")
except Exception as e:
    print(f"⚠ Could not load legal_summarization: {e}")
    ds_legal = None

# 3. Load Wiki Auto (Subset) - To teach "Simplification" mechanics
print("\nDownloading 'wiki_auto' (General Simplification)...")
try:
    # We use the 'manual' subset which is higher quality human simplifications
    ds_wiki = load_dataset("wiki_auto", "manual", split="train")
    
    # Take a sample of 2000 to avoid overwhelming the legal data
    ds_wiki = ds_wiki.shuffle(seed=42).select(range(2000))
    
    # Rename columns (normal_sentence -> src, simple_sentence -> tgt)
    ds_wiki = ds_wiki.rename_column("normal_sentence", "src_legal")
    ds_wiki = ds_wiki.rename_column("simple_sentence", "tgt_plain")
    ds_wiki = ds_wiki.remove_columns([c for c in ds_wiki.column_names if c not in ['src_legal', 'tgt_plain']])
    print(f"✓ Added {len(ds_wiki)} examples from Wiki Auto")
except Exception as e:
    print(f"⚠ Could not load wiki_auto: {e}")
    ds_wiki = None

# 4. Combine Datasets
datasets_to_merge = [d for d in [ds_real, ds_legal, ds_wiki] if d is not None]
if not datasets_to_merge:
    raise ValueError("No datasets loaded! Check your internet connection or CSV path.")

combined_dataset = concatenate_datasets(datasets_to_merge)
print(f"\nTOTAL TRAINING DATASET SIZE: {len(combined_dataset)} pairs")

# 5. Add T5 Prefix
def add_prefix(example):
    example['src_legal'] = 'simplify legal text: ' + str(example['src_legal'])
    return example

combined_dataset = combined_dataset.map(add_prefix)

# 6. Train/Val Split
dataset_split = combined_dataset.train_test_split(test_size=0.1, seed=42)
train_dataset = dataset_split['train']
val_dataset = dataset_split['test']

# --- MODEL SETUP ---
print("\n" + "="*60)
print("MODEL SETUP WITH REGULARIZATION")
print("="*60)

# Load Tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)

# Load Model with DROPOUT (Helps prevent overfitting/copying)
config = T5Config.from_pretrained(MODEL_NAME, dropout_rate=0.15)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)

def preprocess_function(examples):
    inputs = examples['src_legal']
    targets = examples['tgt_plain']
    
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding='max_length')
    
    # Replace pad token ids with -100
    model_inputs['labels'] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in ls]
        for ls in labels["input_ids"]
    ]
    return model_inputs

print("Tokenizing data...")
tokenized_train = train_dataset.map(preprocess_function, batched=True)
tokenized_val = val_dataset.map(preprocess_function, batched=True)

# --- TRAINING ---
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.05,             # Stronger weight decay for regularization
    save_total_limit=2,
    predict_with_generate=True,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss", # Minimize loss to ensure it learns the task
    greater_is_better=False,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

print("\nStarting Training...")
trainer.train()

print("\nSaving Best Model...")
trainer.save_model(str(MODEL_DIR / "best_model"))
tokenizer.save_pretrained(str(MODEL_DIR / "best_model"))
print(f"✓ Saved to {MODEL_DIR / 'best_model'}")