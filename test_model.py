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
    """Translate legal jargon to plain English"""
    inputs = tokenizer(legal_text, return_tensors="pt", max_length=MAX_LENGTH, truncation=True).to(DEVICE)
    
    with torch.no_grad():
        outputs = model.generate(
            **inputs,
            max_length=MAX_LENGTH,
            num_beams=6,  # More beams for better quality
            repetition_penalty=3.0,  # AGGRESSIVE penalty
            length_penalty=1.2,  # Encourage longer paraphrases
            no_repeat_ngram_size=3,  # Prevent 3-gram repetition
            early_stopping=True,
            do_sample=True,  # ADD: Sampling for diversity
            temperature=0.7,  # ADD: Temperature for controlled randomness
            top_p=0.9,  # ADD: Nucleus sampling
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
