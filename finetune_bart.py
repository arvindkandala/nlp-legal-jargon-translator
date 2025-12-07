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
DATA_PATH = Path("data/combined_training_data.csv")  # Uses your cleaned data
MODEL_DIR = Path("models/bart_legal_simplifier")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "facebook/bart-base"
BATCH_SIZE = 8
NUM_EPOCHS = 20  # Reduced slightly as cleaner data converges faster
LEARNING_RATE = 8e-5  # Keep this low for stability

# Load data
print("Loading combined training data...")
df = pd.read_csv(DATA_PATH)
df = df[['src_legal', 'tgt_plain']].dropna()
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

# Training arguments
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    label_smoothing_factor=0.1,
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    predict_with_generate=True,
    fp16=torch.cuda.is_available(),
    report_to="none",
    
    # DIVERSITY SETTINGS (Crucial for preventing copying)
    generation_num_beams=4,
    diversity_penalty=0.3,     # Penalizes repeated beams
    forced_bos_token_id=None,  # Allows flexible start
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=DataCollatorForSeq2Seq(tokenizer, model=model),
)

print("Starting Training...")
trainer.train()
trainer.save_model(str(MODEL_DIR / "best_model"))
tokenizer.save_pretrained(str(MODEL_DIR / "best_model"))
print(f"\n✓ Model saved to {MODEL_DIR / 'best_model'}")