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
MODEL_DIR = Path("models/t5_legal_simplifier")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MODEL_NAME = "t5-base"  # or "t5-small" for faster training
MAX_INPUT_LENGTH = 256
MAX_TARGET_LENGTH = 256
BATCH_SIZE = 8
NUM_EPOCHS = 7  # Early stopping will kick in
LEARNING_RATE = 5e-5

# Check for GPU
if not torch.cuda.is_available():
    print("⚠️  WARNING: No GPU detected. Training will be very slow.")
    print("   In Colab: Runtime → Change runtime type → GPU")
else:
    print(f"✓ GPU detected: {torch.cuda.get_device_name(0)}")

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print(f"Using device: {DEVICE}")

# Load data (ONLY real pairs)
df = pd.read_csv(DATA_PATH)
print(f"Loaded {len(df)} real pairs from {DATA_PATH}")

# Add prefix to source text (T5 convention)
df['src_legal'] = 'simplify legal text: ' + df['src_legal'].astype(str)

# Train/val split
train_df, val_df = train_test_split(df, test_size=0.15, random_state=42)
print(f"Train: {len(train_df)}, Val: {len(val_df)}")

# Convert to HuggingFace datasets
train_dataset = Dataset.from_pandas(train_df[['src_legal', 'tgt_plain']])
val_dataset = Dataset.from_pandas(val_df[['src_legal', 'tgt_plain']])

# Load tokenizer and model
print("\nLoading model and tokenizer...")
tokenizer = T5Tokenizer.from_pretrained(MODEL_NAME, legacy=False)
model = T5ForConditionalGeneration.from_pretrained(MODEL_NAME)
print("✓ Model loaded")

# Preprocessing function
def preprocess_function(examples):
    inputs = examples['src_legal']
    targets = examples['tgt_plain']
    
    # Tokenize inputs
    model_inputs = tokenizer(
        inputs, 
        max_length=MAX_INPUT_LENGTH, 
        truncation=True,
        padding='max_length'
    )
    
    # Tokenize targets
    labels = tokenizer(
        targets, 
        max_length=MAX_TARGET_LENGTH, 
        truncation=True,
        padding='max_length'
    )
    
    # Replace pad token ids with -100 so they're ignored by loss
    model_inputs['labels'] = [
        [(label if label != tokenizer.pad_token_id else -100) for label in labels_example]
        for labels_example in labels["input_ids"]
    ]
    
    return model_inputs

# Tokenize datasets
print("\nTokenizing datasets...")
tokenized_train = train_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=['src_legal', 'tgt_plain']
)
tokenized_val = val_dataset.map(
    preprocess_function, 
    batched=True, 
    remove_columns=['src_legal', 'tgt_plain']
)
print("✓ Tokenization complete")

# Data collator
data_collator = DataCollatorForSeq2Seq(tokenizer, model=model)

# Load evaluation metrics
print("\nLoading evaluation metrics...")
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")
print("✓ Metrics loaded")

def compute_metrics(eval_preds):
    """
    Compute multiple evaluation metrics independent of training loss
    """
    predictions, labels = eval_preds
    
    # Handle case where predictions is a tuple (from generate method)
    if isinstance(predictions, tuple):
        predictions = predictions[0]
    
    # Decode predictions and labels
    decoded_preds = tokenizer.batch_decode(predictions, skip_special_tokens=True)
    
    # Replace -100 in labels (used for padding)
    labels = np.where(labels != -100, labels, tokenizer.pad_token_id)
    decoded_labels = tokenizer.batch_decode(labels, skip_special_tokens=True)
    
    # Clean up text
    decoded_preds = [pred.strip() for pred in decoded_preds]
    decoded_labels = [[label.strip()] for label in decoded_labels]  # BLEU expects list of references
    
    # BLEU score
    bleu_result = bleu_metric.compute(
        predictions=decoded_preds, 
        references=decoded_labels
    )
    
    # ROUGE scores
    rouge_result = rouge_metric.compute(
        predictions=decoded_preds,
        references=[label[0] for label in decoded_labels]
    )
    
    # BERTScore (semantic similarity - holistic evaluation)
    bertscore_result = bertscore_metric.compute(
        predictions=decoded_preds,
        references=[label[0] for label in decoded_labels],
        lang="en",
        model_type="distilbert-base-uncased"
    )
    
    return {
        'bleu': bleu_result['bleu'],
        'rouge1': rouge_result['rouge1'],
        'rouge2': rouge_result['rouge2'],
        'rougeL': rouge_result['rougeL'],
        'bertscore_f1': np.mean(bertscore_result['f1'])
    }

# Training arguments - FIXED for transformers 4.40+
training_args = Seq2SeqTrainingArguments(
    output_dir=MODEL_DIR,
    eval_strategy="epoch",  # Changed from evaluation_strategy
    save_strategy="epoch",
    learning_rate=LEARNING_RATE,
    per_device_train_batch_size=BATCH_SIZE,
    per_device_eval_batch_size=BATCH_SIZE,
    num_train_epochs=NUM_EPOCHS,
    weight_decay=0.01,
    save_total_limit=3,
    predict_with_generate=True,
    generation_max_length=MAX_TARGET_LENGTH,
    logging_dir=str(MODEL_DIR / "logs"),
    logging_steps=20,
    load_best_model_at_end=True,
    metric_for_best_model="bertscore_f1",
    greater_is_better=True,
    fp16=torch.cuda.is_available(),
    push_to_hub=False,
    report_to="none",
)

# Initialize trainer
trainer = Seq2SeqTrainer(
    model=model,
    args=training_args,
    train_dataset=tokenized_train,
    eval_dataset=tokenized_val,
    tokenizer=tokenizer,
    data_collator=data_collator,
    compute_metrics=compute_metrics,
)

# Train the model
print("\n" + "="*60)
print("STARTING TRAINING")
print("="*60 + "\n")

trainer.train()

# Save the best model
print("\nSaving model...")
trainer.save_model(str(MODEL_DIR / "best_model"))
tokenizer.save_pretrained(str(MODEL_DIR / "best_model"))
print(f"✓ Model saved to {MODEL_DIR / 'best_model'}")

# Generate example outputs
print("\n" + "="*60)
print("EXAMPLE OUTPUTS")
print("="*60 + "\n")

model.eval()
model.to(DEVICE)

num_examples = 5
if len(val_df) > 0:
    example_indices = np.random.choice(
        len(val_df), 
        size=min(num_examples, len(val_df)), 
        replace=False
    )
    
    for idx in example_indices:
        source_text = val_df.iloc[idx]['src_legal']
        target_text = val_df.iloc[idx]['tgt_plain']
        
        # Generate prediction
        inputs = tokenizer(
            source_text, 
            return_tensors="pt", 
            max_length=MAX_INPUT_LENGTH, 
            truncation=True
        ).to(DEVICE)
        
        with torch.no_grad():
            outputs = model.generate(
                **inputs,
                max_length=MAX_TARGET_LENGTH,
                num_beams=4,
                early_stopping=True,
                no_repeat_ngram_size=3
            )
        
        prediction = tokenizer.decode(outputs[0], skip_special_tokens=True)
        
        print(f"SOURCE:\n{source_text[25:]}\n")  # Remove prefix
        print(f"TARGET:\n{target_text}\n")
        print(f"PREDICTION:\n{prediction}\n")
        print("-" * 60 + "\n")

print("Training complete!")