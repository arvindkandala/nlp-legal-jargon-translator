import pandas as pd
import torch
from pathlib import Path
from transformers import BartTokenizer, BartForConditionalGeneration
import evaluate
import numpy as np
from tqdm import tqdm

# Configuration
MODEL_PATH = Path("models/bart_legal_simplifier/best_model")
TEST_DATA_PATH = Path("data/real_test_set.csv")
RESULTS_PATH = Path("evaluation_results")
RESULTS_PATH.mkdir(exist_ok=True)

MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from {MODEL_PATH}")
try:
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)  # CHANGED: BartTokenizer
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)  # CHANGED: BartForConditionalGeneration
    model.to(DEVICE)
    model.eval()
    print("✓ Model loaded successfully")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Did you run finetune_bart.py first?")
    exit(1)

# Load held-out test data (100% real)
print(f"\nLoading test data from {TEST_DATA_PATH}")
try:
    test_df = pd.read_csv(TEST_DATA_PATH)
    print(f"✓ Loaded {len(test_df)} real test examples")
except FileNotFoundError:
    print(f"Error: {TEST_DATA_PATH} not found. Run finetune_bart.py first to generate test set.")
    exit(1)

# Load metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

def simplify_text(legal_text):
    # REMOVED instruction prompt - BART learns from examples directly
    inputs = tokenizer(legal_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=6,  # INCREASED from 5
            repetition_penalty=3.0,  # INCREASED from 2.5
            length_penalty=1.2,  # INCREASED from 1.0
            no_repeat_ngram_size=3,  # INCREASED from 2
            early_stopping=True,
            do_sample=True,  # NEW: Enable sampling
            temperature=0.7,  # NEW: Control randomness
            top_p=0.9,  # NEW: Nucleus sampling
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerating predictions on held-out real test set...")
predictions = []
references = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    pred = simplify_text(row['src_legal'])
    predictions.append(pred)
    references.append(row['tgt_plain'])

print("\nComputing evaluation metrics...")
bleu_score = bleu_metric.compute(predictions=predictions, references=[[r] for r in references])
rouge_scores = rouge_metric.compute(predictions=predictions, references=references)
bertscore_results = bertscore_metric.compute(predictions=predictions, references=references, lang="en", model_type="distilbert-base-uncased")

# Print results
print("\n" + "="*70)
print("EVALUATION RESULTS (Real Test Data Only)")
print("="*70)
print(f"Model: BART")
print(f"Test Set Size: {len(test_df)} real pairs")
print(f"\nMetrics:")
print(f"  BLEU Score:    {bleu_score['bleu']:.4f}")
print(f"  ROUGE-L:       {rouge_scores['rougeL']:.4f}")
print(f"  BERTScore F1:  {np.mean(bertscore_results['f1']):.4f}")

# Save detailed results
results_df = pd.DataFrame({
    'source': test_df['src_legal'].values,
    'target': references,
    'prediction': predictions,
    'bertscore_f1': bertscore_results['f1']
})
results_df = results_df.sort_values('bertscore_f1', ascending=False)
results_df.to_csv(RESULTS_PATH / "detailed_results.csv", index=False)
print(f"\n💾 Detailed results saved to {RESULTS_PATH / 'detailed_results.csv'}")

# Show best and worst examples
print("\n" + "="*70)
print("SAMPLE PREDICTIONS (Best 3)")
print("="*70)
for i in range(min(3, len(results_df))):
    row = results_df.iloc[i]
    print(f"\nExample {i+1} (BERTScore: {row['bertscore_f1']:.4f})")
    print(f"Source:     {row['source'][:100]}...")
    print(f"Target:     {row['target'][:100]}...")
    print(f"Prediction: {row['prediction'][:100]}...")

print("\n" + "="*70)
print("SAMPLE PREDICTIONS (Worst 3)")
print("="*70)
for i in range(max(0, len(results_df)-3), len(results_df)):
    row = results_df.iloc[i]
    print(f"\nExample {len(results_df)-i} (BERTScore: {row['bertscore_f1']:.4f})")
    print(f"Source:     {row['source'][:100]}...")
    print(f"Target:     {row['target'][:100]}...")
    print(f"Prediction: {row['prediction'][:100]}...")

print("\n✓ Evaluation complete!")
