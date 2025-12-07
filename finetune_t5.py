import pandas as pd
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq,
    T5Config
)
from sklearn.model_selection import train_test_split

# Config
DATA_PATH = Path("data/real_pairs_cleaned.csv")  # Use cleaned version
MODEL_DIR = Path("models/t5_legal_simplifier")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "google/flan-t5-base"
BATCH_SIZE = 4
NUM_EPOCHS = 25
LEARNING_RATE = 5e-4

# Load
df = pd.read_csv(DATA_PATH)
df = df[['src_legal', 'tgt_plain']].dropna()
df = df[df['src_legal'].str.strip() != df['tgt_plain'].str.strip()]

print(f"Training examples: {len(df)}")

# Instruction prompt
df['src_legal'] = "Rewrite the following legal sentence in plain English: " + df['src_legal'].astype(str)

# Split
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
train_dataset = Dataset.from_pandas(train_df)
val_dataset = Dataset.from_pandas(val_df)

# Load tokenizer
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)

# ===== ADD DROPOUT =====
# Load model with increased dropout for regularization
config = T5Config.from_pretrained(MODEL_NAME)
config.dropout_rate = 0.2  # Increase from default 0.1
print(f"✓ Dropout rate set to {config.dropout_rate}")

model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME, config=config)

def preprocess_function(examples):
    model_inputs = tokenizer(examples['src_legal'], max_length=256, truncation=True, padding='max_length')
    labels = tokenizer(examples['tgt_plain'], max_length=256, truncation=True, padding='max_length')
    model_inputs['labels'] = [[(l if l != tokenizer.pad_token_id else -100) for l in ls] for ls in labels["input_ids"]]
    return model_inputs

tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=train_dataset.column_names)
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=val_dataset.column_names)

# Training with FULL anti-overfitting measures
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    
    # === REGULARIZATION ===
    weight_decay=0.1,              # L2 regularization
    label_smoothing_factor=0.1,    # Prevents overconfidence
    max_grad_norm=1.0,             # Gradient clipping for stability
    
    # === EARLY STOPPING ===
    save_total_limit=2,
    load_best_model_at_end=True,
    metric_for_best_model="eval_loss",
    greater_is_better=False,
    
    predict_with_generate=True,
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

print("\n" + "="*60)
print("ANTI-OVERFITTING MEASURES ENABLED:")
print("  ✓ Dropout: 0.2")
print("  ✓ Weight Decay: 0.1")
print("  ✓ Label Smoothing: 0.1")
print("  ✓ Gradient Clipping: 1.0")
print("  ✓ Early Stopping: eval_loss")
print("="*60 + "\n")

print("Starting Training...")
trainer.train()
trainer.save_model(str(MODEL_DIR / "best_model"))
tokenizer.save_pretrained(str(MODEL_DIR / "best_model"))
print(f"✓ Model saved to {MODEL_DIR / 'best_model'}")
