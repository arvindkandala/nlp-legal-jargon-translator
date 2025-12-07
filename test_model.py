import torch
from pathlib import Path
from transformers import T5Tokenizer, T5ForConditionalGeneration

# Configuration
MODEL_PATH = Path("models/t5_legal_simplifier/best_model")
MAX_LENGTH = 256
DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")

print("="*70)
print("LEGAL JARGON TRANSLATOR - INTERACTIVE MODE")
print("="*70)

print(f"\nLoading model from {MODEL_PATH}...")
try:
    tokenizer = T5Tokenizer.from_pretrained(MODEL_PATH)
    model = T5ForConditionalGeneration.from_pretrained(MODEL_PATH)
    model.to(DEVICE)
    model.eval()
    print("✓ Model loaded successfully\n")
except Exception as e:
    print(f"Error loading model: {e}")
    print("Make sure you've run finetune_t5.py first!")
    exit(1)

def simplify_text(legal_text, num_beams=5):
    """Translate legal jargon to plain English"""
    input_text = f"Rewrite the following legal sentence in plain English: {legal_text}"
    inputs = tokenizer(input_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=num_beams,
            repetition_penalty=2.5,
            length_penalty=1.0,
            no_repeat_ngram_size=2,
            early_stopping=True
        )
    return tokenizer.decode(outputs[0], skip_special_tokens=True)

print("="*70)
print("Enter legal text to simplify (or 'quit' to exit)")
print("="*70 + "\n")

while True:
    user_input = input("Legal text: ").strip()
    
    if user_input.lower() in ['quit', 'exit', 'q']:
        print("\n✓ Goodbye!")
        break
    
    if not user_input:
        print("Please enter some text.\n")
        continue
    
    simplified = simplify_text(user_input)
    print(f"\nSimplified: {simplified}\n")
    print("-" * 70 + "\n")