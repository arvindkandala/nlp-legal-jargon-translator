import torch
from pathlib import Path
from transformers import BartTokenizer, BartForConditionalGeneration

# Configuration
MODEL_PATH = Path("models/bart_legal_simplifier/best_model")
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("LEGAL JARGON TRANSLATOR - INTERACTIVE MODE (BART)")
print("="*70)

print(f"\nLoading model from {MODEL_PATH}...")
try:
    tokenizer = BartTokenizer.from_pretrained(MODEL_PATH)
    model = BartForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("✓ Model loaded successfully\n")
except Exception as e:
    print(f"Error loading model: {e}")
    exit(1)

def simplify_text(legal_text):
    inputs = tokenizer(legal_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            # Balanced Parameters
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

print("Enter legal text to simplify (or 'quit' to exit)")
while True:
    user_input = input("\nLegal text: ").strip()
    if user_input.lower() in ['quit', 'exit', 'q']:
        break
    if not user_input:
        continue
    
    simplified = simplify_text(user_input)
    print(f"\nSimplified: {simplified}")
    print("-" * 70)
