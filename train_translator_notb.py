import csv
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# Settings and hyperparameters
DATA_PATH = Path("data/combined_pairs_no_tb.csv")
MODEL_DIR = Path("models/manual_simplifier_notb")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_FREQ = 2
MAX_SRC_LEN = 80
MAX_TGT_LEN = 80

EMBED_DIM = 128
HIDDEN_DIM = 256
DROPOUT = 0.3
BATCH_SIZE = 16
NUM_EPOCHS = 60 #not all 60 are likely to be run because of early stopping
LEARNING_RATE = .001
CLIP_GRAD = 1.0

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# Tokenization
def tokenize(text: str):
    text = str(text).strip().lower()
    tokens = re.findall(r"\w+|\S", text)
    tokens = [t for t in tokens if t not in {"[", "]"}]
    return tokens

# Load data
df = pd.read_csv(DATA_PATH)
src_list = df["src_legal"].astype(str).tolist()
tgt_list = df["tgt_plain"].astype(str).tolist()

print(f"Loaded {len(src_list)} pairs from {DATA_PATH}")

# Buiild vocabulary
counter = Counter()
for s, t in zip(src_list, tgt_list):
    counter.update(tokenize(s))
    counter.update(tokenize(t))

PAD = "<pad>" #in case you want to pad the sentences to make it fit a specific length
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"

vocab = [PAD, UNK, SOS, EOS]
for tok, freq in counter.most_common():
    if freq >= MIN_FREQ and tok not in vocab:
        vocab.append(tok)

word2idx = {w: i for i, w in enumerate(vocab)}
idx2word = {i: w for w, i in word2idx.items()}

PAD_IDX = word2idx[PAD]
UNK_IDX = word2idx[UNK]
SOS_IDX = word2idx[SOS]
EOS_IDX = word2idx[EOS]

vocab_size = len(vocab)
print("Vocab size:", vocab_size)

def encode(text: str, max_len: int):
    tokens = tokenize(text)
    ids = [word2idx.get(tok, UNK_IDX) for tok in tokens]
    ids = [SOS_IDX] + ids + [EOS_IDX]
    if len(ids) < max_len:
        ids = ids + [PAD_IDX] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        ids[-1] = EOS_IDX
    return ids

# Dataset and DataLoader
class LegalSimpleDataset(Dataset):
    def __init__(self, src_list, tgt_list):
        self.src_list = src_list
        self.tgt_list = tgt_list

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):
        src_ids = encode(self.src_list[idx], MAX_SRC_LEN)
        tgt_ids = encode(self.tgt_list[idx], MAX_TGT_LEN)
        return (
            torch.tensor(src_ids, dtype=torch.long),
            torch.tensor(tgt_ids, dtype=torch.long),
        )

# Split the data
n_total = len(src_list)
n_train = int(0.8 * n_total)
indices = list(range(n_total))
random.seed(42)
random.shuffle(indices)
train_idx = indices[:n_train]
val_idx = indices[n_train:]

train_src = [src_list[i] for i in train_idx]
train_tgt = [tgt_list[i] for i in train_idx]
val_src = [src_list[i] for i in val_idx]
val_tgt = [tgt_list[i] for i in val_idx]

train_data = LegalSimpleDataset(train_src, train_tgt)
val_data = LegalSimpleDataset(val_src, val_tgt)

train_loader = DataLoader(train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False)
val_loader = DataLoader(val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False)

print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")

#attention mechanism
class DotAttention(nn.Module):
    def __init__(self):
        super().__init__()

    def forward(self, hidden, encoder_outputs, src_mask=None):
        """
        hidden state dimension: (B, H)
        encoder_outputs tnesor dimensions: (B, src_len, H)
        src_mask: (B, src_len) - 1 for valid, 0 for padding
        """
        scores = torch.bmm(encoder_outputs, hidden.unsqueeze(2)).squeeze(2)  #(B, src_len)
        
        if src_mask is not None:
            scores = scores.masked_fill(src_mask == 0, -1e9)
        
        attn_weights = torch.softmax(scores, dim=1)
        context = torch.bmm(attn_weights.unsqueeze(1), encoder_outputs).squeeze(1)
        return context, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx, dropout=0.3):
        super().__init__()
        self.pad_idx = pad_idx
        
        self.embedding = nn.Embedding(vocab_size, embed_dim, padding_idx=pad_idx)
        self.embed_dropout = nn.Dropout(dropout)
        
        self.encoder = nn.GRU(embed_dim, hidden_dim, batch_first=True, dropout=dropout if dropout > 0 else 0)
        self.decoder = nn.GRU(embed_dim, hidden_dim, batch_first=True)
        
        self.attn = DotAttention()
        self.dropout = nn.Dropout(dropout)
        self.out = nn.Linear(hidden_dim * 2, vocab_size)

    def create_mask(self, src):
        return (src != self.pad_idx).long()

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        batch_size, tgt_len = tgt.shape
        
        #Encode
        src_mask = self.create_mask(src)
        embedded_src = self.embed_dropout(self.embedding(src))
        encoder_outputs, hidden = self.encoder(embedded_src)
        
        # Decode
        logits = torch.zeros(batch_size, tgt_len, vocab_size, device=src.device)
        input_tok = tgt[:, 0]

        for t in range(1, tgt_len):
            embedded = self.embed_dropout(self.embedding(input_tok)).unsqueeze(1)
            dec_output, hidden = self.decoder(embedded, hidden)
            dec_hidden = hidden[-1]

            context, _ = self.attn(dec_hidden, encoder_outputs, src_mask)
            combined = torch.cat([dec_output.squeeze(1), context], dim=1)
            combined = self.dropout(combined)
            
            step_logits = self.out(combined)
            logits[:, t, :] = step_logits

            teacher_force = random.random() < teacher_forcing_ratio
            top1 = step_logits.argmax(dim=1)
            input_tok = tgt[:, t] if teacher_force else top1

        return logits

    def generate(self, src_text, max_len=MAX_TGT_LEN, repetition_penalty=1.2):
        self.eval()
        with torch.no_grad():
            src_ids = encode(src_text, MAX_SRC_LEN)
            src_tensor = torch.tensor(src_ids, dtype=torch.long, device=DEVICE).unsqueeze(0)
            
            src_mask = self.create_mask(src_tensor)
            embedded_src = self.embedding(src_tensor)
            encoder_outputs, hidden = self.encoder(embedded_src)

            input_tok = torch.tensor([SOS_IDX], dtype=torch.long, device=DEVICE)
            outputs = []
            token_counts = Counter()  #track token usage to see if they get excessively repeated

            for _ in range(max_len):
                embedded = self.embedding(input_tok).unsqueeze(1)
                dec_output, hidden = self.decoder(embedded, hidden)
                dec_hidden = hidden[-1]

                context, _ = self.attn(dec_hidden, encoder_outputs, src_mask)
                combined = torch.cat([dec_output.squeeze(1), context], dim=1)
                step_logits = self.out(combined)
                
                # apply repetition penalty
                for token_id, count in token_counts.items():
                    if count > 0:
                        step_logits[0, token_id] /= (repetition_penalty ** count)
                
                top1 = step_logits.argmax(dim=1)
                token_id = top1.item()
                
                if token_id in (EOS_IDX, PAD_IDX):
                    break
                
                token_counts[token_id] += 1
                outputs.append(idx2word.get(token_id, UNK))
                input_tok = top1

        return " ".join(outputs)


model = Seq2Seq(vocab_size, EMBED_DIM, HIDDEN_DIM, PAD_IDX, DROPOUT).to(DEVICE)
print(model)

#training setup
criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)
scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, mode='min', factor=0.5, patience=2)

def train_epoch(model, loader, epoch):
    model.train()
    total_loss = 0.0
    
    # Decrease teacher forcing as training progresses
    tf_ratio = max(0.5, 0.9 - epoch * 0.05)

    for src_batch, tgt_batch in loader:
        src_batch = src_batch.to(DEVICE)
        tgt_batch = tgt_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(src_batch, tgt_batch, teacher_forcing_ratio=tf_ratio)
        
        B, T, V = logits.shape
        loss = criterion(
            logits[:, 1:, :].reshape(B * (T - 1), V),
            tgt_batch[:, 1:].reshape(B * (T - 1)),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), CLIP_GRAD)
        optimizer.step()

        total_loss += loss.item() * src_batch.size(0)

    return total_loss / len(loader.dataset)


def eval_epoch(model, loader):
    model.eval()
    total_loss = 0.0

    with torch.no_grad():
        for src_batch, tgt_batch in loader:
            src_batch = src_batch.to(DEVICE)
            tgt_batch = tgt_batch.to(DEVICE)

            logits = model(src_batch, tgt_batch, teacher_forcing_ratio=0.0)
            B, T, V = logits.shape
            loss = criterion(
                logits[:, 1:, :].reshape(B * (T - 1), V),
                tgt_batch[:, 1:].reshape(B * (T - 1)),
            )
            total_loss += loss.item() * src_batch.size(0)

    return total_loss / len(loader.dataset)


# Training loop with early stopping in case the validation loss does not go down
best_val_loss = float('inf')
patience = 5
patience_counter = 0

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_epoch(model, train_loader, epoch)
    val_loss = eval_epoch(model, val_loader)
    
    scheduler.step(val_loss)

    print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    #Early stopping
    if val_loss < best_val_loss:
        best_val_loss = val_loss
        patience_counter = 0
        torch.save(model.state_dict(), MODEL_DIR / "best_model.pt")
    else:
        patience_counter += 1
        if patience_counter >= patience:
            print(f"Early stopping at epoch {epoch}")
            break

    if len(val_src) > 0:
        example_text = random.choice(val_src)
        model_out = model.generate(example_text)
        print("  Example src:", example_text[:150], "...")
        print("  Model out  :", model_out)
        print("-" * 70)


torch.save(model.state_dict(), MODEL_DIR / "model.pt")
torch.save(word2idx, MODEL_DIR / "word2idx.pt")
torch.save(idx2word, MODEL_DIR / "idx2word.pt")
print("Saved model and vocab to", MODEL_DIR)