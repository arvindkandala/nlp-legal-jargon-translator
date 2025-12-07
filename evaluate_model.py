import pandas as pd
import torch
from pathlib import Path
from transformers import BartTokenizer, BartForConditionalGeneration
import evaluate
import numpy as np
from tqdm import tqdm
from difflib import SequenceMatcher
import textwrap

# Configuration
MODEL_PATH = Path("models/bart_legal_simplifier/best_model")
TEST_DATA_PATH = Path("data/real_test_set.csv")
RESULTS_PATH = Path("evaluation_results")
RESULTS_PATH.mkdir(exist_ok=True)
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

def print_wrapped(label, text, width=100):
    wrapped = textwrap.fill(str(text), width=width, subsequent_indent='    ')
    print(f"{label}: {wrapped}")

print(f"Loading model from {MODEL_PATH}")
try:
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
except Exception as e:
    print(f"Error: {e}")
    exit(1)

print(f"\nLoading test data from {TEST_DATA_PATH}")
test_df = pd.read_csv(TEST_DATA_PATH)

# Load metrics
bleu = evaluate.load("bleu")
rouge = evaluate.load("rouge")
bertscore = evaluate.load("bertscore")

def similarity_ratio(text1, text2):
    return SequenceMatcher(None, str(text1).lower(), str(text2).lower()).ratio()

def simplify_text(legal_text):
    inputs = tokenizer(legal_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(DEVICE)
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=5,
            repetition_penalty=1.2,
            length_penalty=1.0,
            no_repeat_ngram_size=3,
            early_stopping=True,
            do_sample=True,
            temperature=0.7,
            top_p=0.9,
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("\nGenerating predictions...")
predictions = []
references = []
sources = []

for idx, row in tqdm(test_df.iterrows(), total=len(test_df)):
    pred = simplify_text(row['src_legal'])
    predictions.append(pred)
    references.append(row['tgt_plain'])
    sources.append(row['src_legal'])

print("\nComputing metrics...")
bleu_score = bleu.compute(predictions=predictions, references=[[r] for r in references])
bertscore_res = bertscore.compute(predictions=predictions, references=references, lang="en")

# Copy metrics
copy_scores = []
trans_scores = []
for src, pred in zip(sources, predictions):
    sim = similarity_ratio(src, pred)
    copy_scores.append(sim)
    trans_scores.append(1.0 - sim)

avg_trans = np.mean(trans_scores)

print("\n" + "="*70)
print(f"RESULTS (Avg Transformation: {avg_trans:.4f})")
print(f"BERTScore F1: {np.mean(bertscore_res['f1']):.4f}")
print("="*70)

# Save results
results_df = pd.DataFrame({
    'source': sources,
    'target': references,
    'prediction': predictions,
    'bertscore_f1': bertscore_res['f1'],
    'copy_score': copy_scores,
    'transformation_score': trans_scores
})
results_df.sort_values('transformation_score', ascending=False).to_csv(RESULTS_PATH / "detailed_results.csv", index=False)
