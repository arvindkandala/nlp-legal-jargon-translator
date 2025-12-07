import pandas as pd
import torch
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration
import evaluate
import numpy as np
from tqdm import tqdm
from sklearn.model_selection import train_test_split

# Configuration
# UPDATED: Matches the save path from finetune_t5.py
MODEL_PATH = Path("models/t5_legal_simplifier/best_model")
DATA_PATH = Path("data/real_pairs.csv")
RESULTS_PATH = Path("evaluation_results")
RESULTS_PATH.mkdir(exist_ok=True)

MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print(f"Loading model from {MODEL_PATH}")
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Error loading model: {e}")
    print("Did you run finetune_t5.py first?")
    exit(1)

# Load data
df = pd.read_csv(DATA_PATH)

# UPDATED: Changed test_size to 0.15 to match finetune_t5.py
# This ensures we evaluate on the same validation set used during training
_, test_df = train_test_split(df, test_size=0.15, random_state=42)
print(f"Evaluating on {len(test_df)} test examples")

# Metrics
bleu_metric = evaluate.load("bleu")
rouge_metric = evaluate.load("rouge")
bertscore_metric = evaluate.load("bertscore")

def simplify_text(legal_text, num_beams=4):
    input_text = f"simplify legal text: {legal_text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs, 
            max_length=MAX_LENGTH, 
            num_beams=num_beams, 
            early_stopping=True, 
            no_repeat_ngram_size=3
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerating predictions...")
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

print("\n" + "="*70 + "\nEVALUATION RESULTS\n" + "="*70)
print(f" BLEU Score: {bleu_score['bleu']:.4f}")
print(f" ROUGE-L: {rouge_scores['rougeL']:.4f}")
print(f" BERTScore F1: {np.mean(bertscore_results['f1']):.4f}")

# Save results
results_df = pd.DataFrame({
    'source': test_df['src_legal'].values,
    'target': references,
    'prediction': predictions,
    'bertscore_f1': bertscore_results['f1']
})
results_df.to_csv(RESULTS_PATH / "detailed_results.csv", index=False)
print(f"\n💾 Detailed results saved to {RESULTS_PATH / 'detailed_results.csv'}")

# Interactive testing
print("\n" + "="*70 + "\nINTERACTIVE TESTING\n" + "="*70)
print("Enter legal text to simplify (or 'quit' to exit):\n")
while True:
    user_input = input("Legal text: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    if user_input:
        simplified = simplify_text(user_input)
        print(f"\nSimplified: {simplified}\n")
