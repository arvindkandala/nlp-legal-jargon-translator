import pandas as pd
import numpy as np
import torch
from pathlib import Path
from datasets import Dataset
from transformers import (
    T5Tokenizer,
    T5ForConditionalGeneration,
    Seq2SeqTrainingArguments,
    Seq2SeqTrainer,
    DataCollatorForSeq2Seq
)
import evaluate
from sklearn.model_selection import train_test_split

# Configuration
DATA_PATH = Path("data/real_pairs.csv")
# UPDATED: Relative path ensures it saves inside your project folder
MODEL_DIR = Path("models/t5_legal_simplifier")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "t5-base"
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 25
LEARNING_RATE = 5e-5

# Check for GPU
if not torch.cuda.is_available():
    print(" WARNING: No GPU detected. Training will be very slow.")
    print(" In Colab: Runtime -> Change runtime type -> GPU")
else:
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load data
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} real pairs from {DATA_PATH}")

# Add prefix (T5 convention)
df['src_legal'] = 'simplify legal text: ' + df['src_legal'].astype(str)

# Train/val split
# NOTE: We use 0.15 here. Evaluate script must match this to avoid data leakage.
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df[['src_legal', 'tgt_plain']])
val_dataset = Dataset.from_pandas(val_df[['src_legal', 'tgt_plain']])

# Load tokenizer and model
print("\nLoading model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)

# Preprocessing
def preprocess_function(examples):
    inputs = examples['src_legal']
    targets = examples['tgt_plain']
    
    model_inputs = tokenizer(inputs, max_length=MAX_INPUT_LENGTH, truncation=True, padding='max_length')
    labels = tokenizer(targets, max_length=MAX_TARGET_LENGTH, truncation=True, padding='max_length')
    
    # Replace pad token ids with -100 so they are ignored by loss
    model_inputs['labels'] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
        for labels_example in labels["input_ids"]
    ]
    return model_inputs

print("\nTokenizing datasets...")
tokenized_train = train_dataset.map(preprocess_function, batched=True, remove_columns=['src_legal', 'tgt_plain'])
tokenized_val = val_dataset.map(preprocess_function, batched=True, remove_columns=['src_legal', 'tgt_plain'])

data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

def compute_metrics(eval_preds):
    predictions, labels = eval_preds
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    predictions = np.where(predictions >= 0, predictions, tokenizer.pad_token_id)
    predictions = np.where(predictions < len(tokenizer), predictions, tokenizer.pad_token_id)
    
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]
    
    bleu = bleu_metric.compute(predictions=decoded_preds, references=decoded_labels)
    rouge = rouge_metric.compute(predictions=decoded_preds, references=[l[0] for l in decoded_labels])
    bertscore = bertscore_metric.compute(predictions=decoded_preds, references=[l[0] for l in decoded_labels], lang="en", model_type="distilbert-base-uncased")
    
    return {
        'bleu': bleu['bleu'],
        'rouge1': rouge['rouge1'],
        'rougeL': rouge['rougeL'],
        'bertscore_f1': np.mean(bertscore['f1'])
    }

training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    save_total_limit=2,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    load_best_model_at_end=True,
    metric_for_best_model="bertscore_f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    report_to="none"
)

trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics
)

print("\n" + "="*60 + "\nSTARTING TRAINING\n" + "="*60)
trainer.train()

print("\nSaving model...")
# Saves to models/t5_legal_simplifier/best_model
trainer.save_model(str(MODEL_DIR / "best_model"))
tokenizer.save_pretrained(str(MODEL_DIR / "best_model"))
print(f"✓ Model saved to {MODEL_DIR / 'best_model'}")
