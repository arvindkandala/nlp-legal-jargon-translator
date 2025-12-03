import pandas as pd
import torch
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate
import numpy as np
from tqdm import tqdm

# Configuration
MODEL_PATH = Path("models/t5_legal_simplifier/best_model")
DATA_PATH = Path("data/real_pairs.csv")
RESULTS_PATH = Path("evaluation_results")
RESULTS_PATH.mkdir(exist_ok=True)

MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from {MODEL_PATH}")
tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
model.to(DEVICE)
model.eval()

# Load test data (use a held-out test set or validation set)
df = pd.read_csv(DATA_PATH)
from sklearn.model_selection import train_test_split
_, test_df = train_test_split(df, test_size=0.1765, random_state=42)

print(f"Evaluating on {len(test_df)} test examples")

# Load metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

def simplify_text(legal_text, num_beams=4):
    """Generate simplified version of legal text"""
    input_text = f"simplify legal text: {legal_text}"
    inputs = tokenizer(
        input_text, 
        return_tensors="pt", 
        max_length=MAX_LENGTH, 
        truncation=True
    ).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=num_beams,
            early_stopping=True,
            no_repeat_ngram_size=3,
            repetition_penalty=1.2
        )
    
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

# Generate predictions for all test examples
print("\nGenerating predictions...")
predictions = []
references = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    pred = simplify_text(row['src_legal'])
    predictions.append(pred)
    references.append(row['tgt_plain'])

# Compute comprehensive metrics
print("\nComputing evaluation metrics...")

# BLEU Score
bleu_score = bleu_metric.compute(
    predictions=predictions,
    references=[[ref] for ref in references]
)

# ROUGE Scores
rouge_scores = rouge_metric.compute(
    predictions=predictions,
    references=references
)

# BERTScore (semantic similarity - holistic evaluation)
bertscore_results = bertscore_metric.compute(
    predictions=predictions,
    references=references,
    lang="en",
    model_type="distilbert-base-uncased"
)

# Additional custom metrics
def compute_length_stats(texts):
    lengths = [len(text.split()) for text in texts]
    return {
        'mean': np.mean(lengths),
        'std': np.std(lengths),
        'median': np.median(lengths)
    }

source_lengths = compute_length_stats(test_df['src_legal'].tolist())
target_lengths = compute_length_stats(references)
pred_lengths = compute_length_stats(predictions)

# Print comprehensive results
print("\n" + "="*70)
print("EVALUATION RESULTS")
print("="*70)

print(f"\n📊 Dataset Statistics:")
print(f"  Test examples: {len(test_df)}")
print(f"  Source length: {source_lengths['mean']:.1f} ± {source_lengths['std']:.1f} words")
print(f"  Target length: {target_lengths['mean']:.1f} ± {target_lengths['std']:.1f} words")
print(f"  Prediction length: {pred_lengths['mean']:.1f} ± {pred_lengths['std']:.1f} words")

print(f"\n📈 Translation Quality Metrics:")
print(f"  BLEU Score: {bleu_score['bleu']:.4f}")
print(f"    - BLEU-1: {bleu_score['precisions'][0]:.4f}")
print(f"    - BLEU-2: {bleu_score['precisions'][1]:.4f}")
print(f"    - BLEU-3: {bleu_score['precisions'][2]:.4f}")
print(f"    - BLEU-4: {bleu_score['precisions'][3]:.4f}")

print(f"\n  ROUGE Scores:")
print(f"    - ROUGE-1: {rouge_scores['rouge1']:.4f}")
print(f"    - ROUGE-2: {rouge_scores['rouge2']:.4f}")
print(f"    - ROUGE-L: {rouge_scores['rougeL']:.4f}")

print(f"\n  BERTScore (Semantic Similarity):")
print(f"    - Precision: {np.mean(bertscore_results['precision']):.4f}")
print(f"    - Recall: {np.mean(bertscore_results['recall']):.4f}")
print(f"    - F1: {np.mean(bertscore_results['f1']):.4f}")

# Save detailed results
results_df = pd.DataFrame({
    'source': test_df['src_legal'].values,
    'target': references,
    'prediction': predictions,
    'bertscore_f1': bertscore_results['f1']
})

# Sort by BERTScore to see best/worst examples
results_df = results_df.sort_values('bertscore_f1', ascending=False)
results_df.to_csv(RESULTS_PATH / "detailed_results.csv", index=False)

print(f"\n💾 Detailed results saved to {RESULTS_PATH / 'detailed_results.csv'}")

# Show best and worst examples for qualitative analysis
print("\n" + "="*70)
print("BEST EXAMPLES (Highest BERTScore)")
print("="*70)

for idx in range(min(3, len(results_df))):
    row = results_df.iloc[idx]
    print(f"\nExample {idx+1} (BERTScore F1: {row['bertscore_f1']:.4f})")
    print(f"\nSOURCE:\n{row['source']}\n")
    print(f"TARGET:\n{row['target']}\n")
    print(f"PREDICTION:\n{row['prediction']}\n")
    print("-" * 70)

print("\n" + "="*70)
print("WORST EXAMPLES (Lowest BERTScore)")
print("="*70)

for idx in range(max(0, len(results_df)-3), len(results_df)):
    row = results_df.iloc[idx]
    print(f"\nExample (BERTScore F1: {row['bertscore_f1']:.4f})")
    print(f"\nSOURCE:\n{row['source']}\n")
    print(f"TARGET:\n{row['target']}\n")
    print(f"PREDICTION:\n{row['prediction']}\n")
    print("-" * 70)

# Interactive testing
print("\n" + "="*70)
print("INTERACTIVE TESTING")
print("="*70)
print("Enter legal text to simplify (or 'quit' to exit):\n")

while True:
    user_input = input("Legal text: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    if user_input:
        simplified = simplify_text(user_input)
        print(f"\nSimplified: {simplified}\n")

print("\n✓ Evaluation complete!")