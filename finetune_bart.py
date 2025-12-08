import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    BartTokenizer,
    BartForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
)
from sklearn.model_selection import train_test_split

# ==========================================
# CONFIGURATION
# ==========================================
DATA_PATH = Path("data/combined_training_data.csv")
MODEL_DIR = Path("models/bart_legal_simplifier")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/bart-base"
BATCH_SIZE = 8
NUM_EPOCHS = 30         # Clean data converges faster
LEARNING_RATE = 7e-5    # OPTIMIZED: The "Goldilocks" rate for BART fine-tuning

# ==========================================
# DATA LOADING
# ==========================================
print("Loading combined training data...")
df = pd.read_csv(DATA_PATH)
df = df[['src_legal', 'tgt_plain']].dropna()

# Basic filtering to ensure no empty strings
df = df[df['src_legal'].str.strip().astype(bool)]
df = df[df['tgt_plain'].str.strip().astype(bool)]

print(f"Total dataset: {len(df)} pairs")

# Split Data
# 1. Isolate Real Data (first 600 rows)
REAL_DATA_COUNT = 600
df_real = df.iloc[:REAL_DATA_COUNT].copy()
df_synthetic = df.iloc[REAL_DATA_COUNT:].copy()

print(f"  Real data: {len(df_real)} pairs")
print(f"  Synthetic data: {len(df_synthetic)} pairs")

# 2. Create Held-out Test Set (Real data only)
real_trainval, real_test = train_test_split(df_real, test_size=0.15, random_state=42)
real_test.to_csv('data/real_test_set.csv', index=False)
print(f"\n✓ Saved {len(real_test)} real pairs for testing")

# 3. Combine remaining Real + Synthetic for training
trainval_combined = pd.concat([real_trainval, df_synthetic], ignore_index=True)
train_df, val_df = train_test_split(trainval_combined, test_size=0.15, random_state=42)

print(f"  Training: {len(train_df)} pairs")
print(f"  Validation: {len(val_df)} pairs")

train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# ==========================================
# MODEL SETUP
# ==========================================
tokenizer = BartTokenizer.from_pretrained(MODEL_NAME)
model = BartForConditionalGeneration.from_pretrained(MODEL_NAME)

def preprocess_function(examples):
    model_inputs = tokenizer(
        examples['src_legal'], 
        max_length=256, 
        truncation=True, 
        padding='max_length'
    )
    labels = tokenizer(
        examples['tgt_plain'], 
        max_length=256, 
        truncation=True, 
        padding='max_length'
    )
    # Replace padding token id with -100 to ignore loss on padding
    model_inputs['labels'] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in ls] 
        for ls in labels["input_ids"]
    ]
    return model_inputs

print("Tokenizing data...")
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

# ==========================================
# TRAINING ARGUMENTS
# ==========================================
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    
    # Regularization
    weight_decay=0.01,
    label_smoothing_factor=0.1,  # OPTIMIZED: Standard 0.1 allows reasonable confidence
    max_grad_norm=1.0,           # Standard gradient clipping
    warmup_steps=500,            # Prevents initial shock
    
    # Model Saving
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    
    # DIVERSITY (Validation only)
    # This helps the "best model" selection favor one that doesn't just copy
    generation_num_beams=4,
    diversity_penalty=0.3,
    forced_bos_token_id=None,
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

# Save final
trainer.save_model(str(MODEL_DIR / "best_model"))
tokenizer.save_pretrained(str(MODEL_DIR / "best_model"))
print(f"\n✓ Model saved to {MODEL_DIR / 'best_model'}")