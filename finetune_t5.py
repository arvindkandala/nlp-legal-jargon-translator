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

# Config
DATA_PATH = Path("data/combined_training_data.csv")
MODEL_DIR = Path("models/bart_legal_simplifier")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/bart-base"  # CHANGED: BART instead of T5
BATCH_SIZE = 8
NUM_EPOCHS = 30  # INCREASED: More epochs like your attention model
LEARNING_RATE = 5e-4  # LOWER: BART needs gentler fine-tuning

# Load data
print("Loading combined training data...")
df = pd.read_csv(DATA_PATH)
df = df[['src_legal', 'tgt_plain']].dropna()
df = df[df['src_legal'].str.strip() != df['tgt_plain'].str.strip()]

print(f"Total dataset: {len(df)} pairs")

# First 600 rows are REAL data
REAL_DATA_COUNT = 600
df_real = df.iloc[:REAL_DATA_COUNT].copy()
df_synthetic = df.iloc[REAL_DATA_COUNT:].copy()

print(f"  Real data: {len(df_real)} pairs")
print(f"  Synthetic data: {len(df_synthetic)} pairs")

# Hold out test set
real_trainval, real_test = train_test_split(df_real, test_size=0.15, random_state=42)
real_test.to_csv('data/real_test_set.csv', index=False)
print(f"\n✓ Saved {len(real_test)} real pairs for testing")

# Combine for training
trainval_combined = pd.concat([real_trainval, df_synthetic], ignore_index=True)
train_df, val_df = train_test_split(trainval_combined, test_size=0.15, random_state=42)

print(f"\nFinal split:")
print(f"  Training: {len(train_df)} pairs")
print(f"  Validation: {len(val_df)} pairs")
print(f"  Testing: {len(real_test)} pairs (100% real)")

# NO instruction prompt for BART - it learns from examples directly
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Model setup
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
    
    model_inputs['labels'] = [
        [(l if l != tokenizer.pad_token_id else -100) for l in ls] 
        for ls in labels["input_ids"]
    ]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

# Training arguments - aggressive settings to force transformation
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    
    # Aggressive regularization to prevent copying
    weight_decay=0.01,
    label_smoothing_factor=0.15,  # INCREASED
    max_grad_norm=0.5,  # TIGHTER clipping
    
    # Warmup to prevent early convergence to copying
    warmup_steps=500,
    
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    
    # Learning rate scheduling
    lr_scheduler_type="reduce_lr_on_plateau",  # Like your attention model
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

print("\n" + "="*60)
print("BART TRAINING CONFIGURATION:")
print(f"  Model: {MODEL_NAME}")
print(f"  Learning Rate: {LEARNING_RATE}")
print(f"  Batch Size: {BATCH_SIZE}")
print(f"  Epochs: {NUM_EPOCHS}")
print(f"  Label Smoothing: 0.15")
print(f"  Weight Decay: 0.01")
print("="*60 + "\n")

print("Starting Training...")
trainer.train()
trainer.save_model(str(MODEL_DIR / "best_model"))
tokenizer.save_pretrained(str(MODEL_DIR / "best_model"))
print(f"\n✓ Model saved to {MODEL_DIR / 'best_model'}")
