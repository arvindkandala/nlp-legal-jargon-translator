from pathlib import Path
from collections import Counter
import random

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

# define parameters

DATA_PATH = Path("data/simple_pairs.csv")
MAX_VOCAB_SIZE = 10000
MAX_SRC_LEN = 60     # max tokens for legal sentence
MAX_TGT_LEN = 60     # max tokens for plain English
BATCH_SIZE = 32
EMB_DIM = 128
HIDDEN_DIM = 256
NUM_EPOCHS = 5
LEARNING_RATE = 1e-3

device = torch.device("cuda" if torch.cuda.is_available()
                      else "mps" if torch.backends.mps.is_available()
                      else "cpu")
print("Using device:", device)


 #1) load the data

df = pd.read_csv(DATA_PATH).dropna()
src_texts = df["src_legal"].astype(str).tolist()
tgt_texts = df["tgt_plain"].astype(str).tolist()

print("Loaded", len(src_texts), "pairs")


# 2) tokenization and vocabulary building

def tokenize(text: str):
    """Very simple tokenizer: lowercase + split on spaces."""
    return text.lower().strip().split()


# build word frequency counter over both src and tgt
counter = Counter()
for s in src_texts + tgt_texts:
    counter.update(tokenize(s))

# special tokens
PAD_TOKEN = "<pad>" # padding token in case sentence too short
SOS_TOKEN = "<sos>" # start of sentence
EOS_TOKEN = "<eos>" # end of sentence
UNK_TOKEN = "<unk>" # unknown token for words not in vocab

special_tokens = [PAD_TOKEN, SOS_TOKEN, EOS_TOKEN, UNK_TOKEN]
word2idx = {tok: i for i, tok in enumerate(special_tokens)}

# Add most common words up to MAX_VOCAB_SIZE
for word, freq in counter.most_common(MAX_VOCAB_SIZE - len(special_tokens)):
    if word not in word2idx:
        word2idx[word] = len(word2idx)

idx2word = {i: w for w, i in word2idx.items()}

PAD_IDX = word2idx[PAD_TOKEN]
SOS_IDX = word2idx[SOS_TOKEN]
EOS_IDX = word2idx[EOS_TOKEN]
UNK_IDX = word2idx[UNK_TOKEN]

vocab_size = len(word2idx)
print("Vocab size:", vocab_size)


def encode(text: str, max_len: int):
    """Turn text into a list of token IDs with <sos>, <eos>, padding."""
    tokens = tokenize(text)
    ids = [word2idx.get(tok, UNK_IDX) for tok in tokens]
    ids = [SOS_IDX] + ids + [EOS_IDX]
    # pad or truncate
    if len(ids) < max_len:
        ids = ids + [PAD_IDX] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        ids[-1] = EOS_IDX  # ensure last token is EOS if truncated
    return ids


# 3) data set and load 

class LegalSimpleDataset(Dataset):
    def __init__(self, src_list, tgt_list):
        self.src_list = src_list
        self.tgt_list = tgt_list

    def __len__(self):
        return len(self.src_list)

    def __getitem__(self, idx):
        src = encode(self.src_list[idx], MAX_SRC_LEN)
        tgt = encode(self.tgt_list[idx], MAX_TGT_LEN)
        return (
            torch.tensor(src, dtype=torch.long),
            torch.tensor(tgt, dtype=torch.long),
        )


# simple train/val split
data = list(zip(src_texts, tgt_texts))
random.seed(42)
random.shuffle(data)
split_idx = int(0.9 * len(data))
train_pairs = data[:split_idx]
val_pairs = data[split_idx:]

train_src, train_tgt = zip(*train_pairs)
val_src, val_tgt = zip(*val_pairs)

train_dataset = LegalSimpleDataset(train_src, train_tgt)
val_dataset = LegalSimpleDataset(val_src, val_tgt)

train_loader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
val_loader = DataLoader(val_dataset, batch_size=BATCH_SIZE)


# 4. model definition 

class Seq2Seq(nn.Module):
    """
    Encoder-decoder with shared embedding and GRU.
    Very simple: no attention, greedy decoding.
    """

    def __init__(self, vocab_size, emb_dim, hidden_dim, pad_idx):
        super().__init__()
        self.vocab_size = vocab_size
        self.hidden_dim = hidden_dim

        self.embedding = nn.Embedding(vocab_size, emb_dim, padding_idx=pad_idx) #128 dimesion imbedding vector

        self.encoder = nn.GRU( #simpler version of LSTM with only update and reset gates 
            emb_dim, hidden_dim, batch_first=True
        )  # input: (B, src_len, emb_dim)

        self.decoder = nn.GRU(
            emb_dim, hidden_dim, batch_first=True
        )  # input: (B, 1, emb_dim) step by step

        self.out = nn.Linear(hidden_dim, vocab_size)

    def forward(self, src, tgt, teacher_forcing_ratio=0.5):
        """
        src: (B, src_len)
        tgt: (B, tgt_len)
        Returns logits of shape (B, tgt_len, vocab_size)
        """
        batch_size, tgt_len = tgt.shape

        # Encoder 
        embedded_src = self.embedding(src)  # (B, src_len, E)
        _, hidden = self.encoder(embedded_src)  # hidden: (1, B, H)

        # Decoder
        outputs = torch.zeros(
            batch_size, tgt_len, self.vocab_size, device=src.device
        )

        # first input to decoder = <sos> for every example
        input_tok = torch.full(
            (batch_size,), SOS_IDX, dtype=torch.long, device=src.device
        )  # (B,)

        for t in range(tgt_len):
            # embed current input token
            embedded = self.embedding(input_tok).unsqueeze(1)  # (B, 1, E)

            # one GRU step
            output, hidden = self.decoder(embedded, hidden)  # output: (B, 1, H)
            logits = self.out(output.squeeze(1))  # (B, vocab_size)
            outputs[:, t, :] = logits

            # decide next input token
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = logits.argmax(dim=1)  # (B,)

            if teacher_force:
                # use ground-truth token at this time step
                input_tok = tgt[:, t]
            else:
                # use model prediction
                input_tok = top1

        return outputs

    def generate(self, src, max_len=MAX_TGT_LEN):
        self.eval()
        with torch.no_grad():
            src = src.unsqueeze(0)  # (1, src_len)
            embedded_src = self.embedding(src)
            _, hidden = self.encoder(embedded_src)

            input_tok = torch.tensor([SOS_IDX], device=src.device)
            outputs = []

            for _ in range(max_len):
                embedded = self.embedding(input_tok).unsqueeze(0)  # (1,1,E)
                output, hidden = self.decoder(embedded, hidden)
                logits = self.out(output.squeeze(1))  # (1,V)
                top1 = logits.argmax(dim=1)  # (1,)
                token_id = top1.item()
                if token_id == EOS_IDX:
                    break
                outputs.append(token_id)
                input_tok = top1

        # convert IDs back to words
        tokens = [idx2word.get(i, UNK_TOKEN) for i in outputs]
        return " ".join(tokens)


model = Seq2Seq(vocab_size, EMB_DIM, HIDDEN_DIM, PAD_IDX).to(device)
print(model)


# training setup

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_epoch():
    model.train()
    total_loss = 0.0
    for src_batch, tgt_batch in train_loader:
        src_batch = src_batch.to(device)
        tgt_batch = tgt_batch.to(device)

        # model outputs: (B, tgt_len, vocab_size)
        logits = model(src_batch, tgt_batch, teacher_forcing_ratio=0.5)

        # reshape for loss: (B * tgt_len, vocab_size) vs (B * tgt_len)
        B, T, V = logits.shape
        loss = criterion(
            logits.view(B * T, V),
            tgt_batch.view(B * T),
        )

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_loss += loss.item()

    return total_loss / len(train_loader)


def eval_epoch():
    model.eval()
    total_loss = 0.0
    with torch.no_grad():
        for src_batch, tgt_batch in val_loader:
            src_batch = src_batch.to(device)
            tgt_batch = tgt_batch.to(device)

            logits = model(src_batch, tgt_batch, teacher_forcing_ratio=0.0)
            B, T, V = logits.shape
            loss = criterion(
                logits.view(B * T, V),
                tgt_batch.view(B * T),
            )
            total_loss += loss.item()
    return total_loss / len(val_loader)


# train loop 

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_epoch()
    val_loss = eval_epoch()
    print(f"Epoch {epoch}: train_loss={train_loss:.4f}  val_loss={val_loss:.4f}")

    # quick sanity check on one example
    example_src = encode(src_texts[0], MAX_SRC_LEN)
    gen = model.generate(torch.tensor(example_src, dtype=torch.long, device=device))
    print("  Example src:", src_texts[0])
    print("  Model out :", gen)
    print("-" * 80)


# Save model

SAVE_DIR = Path("models/manual_simplifier")
SAVE_DIR.mkdir(parents=True, exist_ok=True)
torch.save(model.state_dict(), SAVE_DIR / "model.pt")
torch.save(word2idx, SAVE_DIR / "word2idx.pt")
torch.save(idx2word, SAVE_DIR / "idx2word.pt")

print("Saved model and vocab to", SAVE_DIR)
