import csv
import random
import re
from collections import Counter
from pathlib import Path

import pandas as pd
import torch
import torch.nn as nn
from torch.utils.data import Dataset, DataLoader

#SETTINGS & HYPERPARAMS

DATA_PATH = Path("data/simple_pairs.csv")   # src_legal, tgt_plain
MODEL_DIR = Path("models/manual_simplifier")
MODEL_DIR.mkdir(parents=True, exist_ok=True)

MIN_FREQ = 2          # minimum word frequency to keep in vocab
MAX_SRC_LEN = 80      # max tokens for source sentences
MAX_TGT_LEN = 80      # max tokens for target sentences

EMBED_DIM = 128
HIDDEN_DIM = 256
BATCH_SIZE = 16
NUM_EPOCHS = 5        # keep small for CPU, can increase later
LEARNING_RATE = 1e-3

DEVICE = torch.device("cuda" if torch.cuda.is_available() else "cpu")
print("Using device:", DEVICE)

# LOAD DATA & BUILD VOCAB


def tokenize(text: str):
    """
    Very simple tokenizer: lowercase and split on words / punctuation.
    """
    text = str(text).strip().lower()
    # words or individual non-whitespace characters
    return re.findall(r"\w+|\S", text)


df = pd.read_csv(DATA_PATH)
src_list = df["src_legal"].astype(str).tolist()
tgt_list = df["tgt_plain"].astype(str).tolist()

print(f"Loaded {len(src_list)} pairs from {DATA_PATH}")

# collect token counts from both source and target
counter = Counter()
for s, t in zip(src_list, tgt_list):
    counter.update(tokenize(s))
    counter.update(tokenize(t))

# special tokens
PAD = "<pad>"
UNK = "<unk>"
SOS = "<sos>"
EOS = "<eos>"

# start vocab with specials
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
    """
    Turn text into a fixed-length list of token IDs:
      [<sos>, w1, w2, ..., <eos>, <pad>, ...]
    """
    tokens = tokenize(text)
    ids = [word2idx.get(tok, UNK_IDX) for tok in tokens]
    ids = [SOS_IDX] + ids + [EOS_IDX]
    if len(ids) < max_len:
        ids = ids + [PAD_IDX] * (max_len - len(ids))
    else:
        ids = ids[:max_len]
        ids[-1] = EOS_IDX  # ensure we end with EOS if truncated
    return ids


# DATASET & DATALOADERS 


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


# simple 80/20 split
n_total = len(src_list)
n_train = int(0.8 * n_total)
indices = list(range(n_total))
random.shuffle(indices)
train_idx = indices[:n_train]
val_idx = indices[n_train:]

train_src = [src_list[i] for i in train_idx]
train_tgt = [tgt_list[i] for i in train_idx]
val_src = [src_list[i] for i in val_idx]
val_tgt = [tgt_list[i] for i in val_idx]

train_data = LegalSimpleDataset(train_src, train_tgt)
val_data = LegalSimpleDataset(val_src, val_tgt)

train_loader = DataLoader(
    train_data, batch_size=BATCH_SIZE, shuffle=True, drop_last=False
)
val_loader = DataLoader(
    val_data, batch_size=BATCH_SIZE, shuffle=False, drop_last=False
)

print(f"Train size: {len(train_data)}, Val size: {len(val_data)}")


# ATTENTION-BASED SEQ2SEQ MODEL 


class DotAttention(nn.Module):
    """
    Luong dot-product attention.
    """

    def __init__(self):
        super().__init__()

    def forward(self, hidden, encoder_outputs):
        """
        hidden: (B, H)       - current decoder hidden state
        encoder_outputs: (B, src_len, H) - all encoder outputs
        Returns:
          context: (B, H)    - weighted sum of encoder_outputs
          attn_weights: (B, src_len)
        """
        # scores: (B, src_len)
        scores = torch.bmm(
            encoder_outputs, hidden.unsqueeze(2)
        ).squeeze(2)

        attn_weights = torch.softmax(scores, dim=1)
        # context: (B, 1, H) -> (B, H)
        context = torch.bmm(
            attn_weights.unsqueeze(1), encoder_outputs
        ).squeeze(1)
        return context, attn_weights


class Seq2Seq(nn.Module):
    def __init__(self, vocab_size, embed_dim, hidden_dim, pad_idx):
        super().__init__()
        self.embedding = nn.Embedding(
            vocab_size, embed_dim, padding_idx=pad_idx
        )
        self.encoder = nn.GRU(
            embed_dim, hidden_dim, batch_first=True
        )
        self.decoder = nn.GRU(
            embed_dim, hidden_dim, batch_first=True
        )
        self.attn = DotAttention()
        self.out = nn.Linear(hidden_dim * 2, vocab_size)

    def forward(self, src, tgt, teacher_forcing_ratio=0.8):
        """
        src: (B, src_len)
        tgt: (B, tgt_len)  -- contains <sos> and <eos>
        Returns:
          logits: (B, tgt_len, vocab_size)
        """
        batch_size, tgt_len = tgt.shape

        # 1) encode source
        embedded_src = self.embedding(src)  # (B, src_len, E)
        encoder_outputs, hidden = self.encoder(embedded_src)
        # hidden: (1, B, H) because 1-layer GRU

        # 2) decode with attention
        logits = torch.zeros(
            batch_size, tgt_len, vocab_size, device=src.device
        )

        # first decoder input is <sos> for every example
        input_tok = tgt[:, 0]  # (B,)

        for t in range(1, tgt_len):
            embedded = self.embedding(input_tok).unsqueeze(1)  # (B, 1, E)
            dec_output, hidden = self.decoder(embedded, hidden)
            # dec_output: (B, 1, H)
            dec_hidden = hidden[-1]  # (B, H)

            context, _ = self.attn(dec_hidden, encoder_outputs)  # (B, H)
            combined = torch.cat(
                [dec_output.squeeze(1), context], dim=1
            )  # (B, 2H)

            step_logits = self.out(combined)  # (B, V)
            logits[:, t, :] = step_logits

            # choose next input token
            teacher_force = random.random() < teacher_forcing_ratio
            top1 = step_logits.argmax(dim=1)  # (B,)
            input_tok = tgt[:, t] if teacher_force else top1

        return logits

    def generate(self, src_text, max_len=MAX_TGT_LEN):
        """
        Greedy decoding with attention.
        """
        self.eval()
        with torch.no_grad():
            src_ids = encode(src_text, MAX_SRC_LEN)
            src_tensor = torch.tensor(
                src_ids, dtype=torch.long, device=DEVICE
            ).unsqueeze(0)  # (1, src_len)

            embedded_src = self.embedding(src_tensor)
            encoder_outputs, hidden = self.encoder(embedded_src)

            input_tok = torch.tensor(
                [SOS_IDX], dtype=torch.long, device=DEVICE
            )
            outputs = []

            for _ in range(max_len):
                embedded = self.embedding(input_tok).unsqueeze(1)  # (1,1,E)
                dec_output, hidden = self.decoder(embedded, hidden)
                dec_hidden = hidden[-1]  # (1, H)

                context, _ = self.attn(dec_hidden, encoder_outputs)
                combined = torch.cat(
                    [dec_output.squeeze(1), context], dim=1
                )  # (1, 2H)
                step_logits = self.out(combined)  # (1, V)
                top1 = step_logits.argmax(dim=1)  # (1,)

                token_id = top1.item()
                if token_id in (EOS_IDX, PAD_IDX):
                    break
                outputs.append(idx2word.get(token_id, UNK))
                input_tok = top1

        return " ".join(outputs)


model = Seq2Seq(vocab_size, EMBED_DIM, HIDDEN_DIM, PAD_IDX).to(DEVICE)
print(model)

# TRAINING & EVAL LOOPS 

criterion = nn.CrossEntropyLoss(ignore_index=PAD_IDX)
optimizer = torch.optim.Adam(model.parameters(), lr=LEARNING_RATE)


def train_epoch(model, loader):
    model.train()
    total_loss = 0.0

    for src_batch, tgt_batch in loader:
        src_batch = src_batch.to(DEVICE)
        tgt_batch = tgt_batch.to(DEVICE)

        optimizer.zero_grad()
        logits = model(src_batch, tgt_batch, teacher_forcing_ratio=0.8)
        # ignore first time step (where target is <sos>)
        B, T, V = logits.shape
        loss = criterion(
            logits[:, 1:, :].reshape(B * (T - 1), V),
            tgt_batch[:, 1:].reshape(B * (T - 1)),
        )
        loss.backward()
        nn.utils.clip_grad_norm_(model.parameters(), 1.0)
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


#  MAIN TRAINING LOOP 

for epoch in range(1, NUM_EPOCHS + 1):
    train_loss = train_epoch(model, train_loader)
    val_loss = eval_epoch(model, val_loader)

    print(
        f"Epoch {epoch}: train_loss={train_loss:.4f}  "
        f"val_loss={val_loss:.4f}"
    )

    # print a random example from validation set to see behavior
    if len(val_src) > 0:
        example_text = random.choice(val_src)
        model_out = model.generate(example_text)
        print("  Example src:", example_text[:200], "...")
        print("  Model out :", model_out)
        print("-" * 70)

# SAVE MODEL & VOCAB 

torch.save(model.state_dict(), MODEL_DIR / "model.pt")
torch.save(word2idx, MODEL_DIR / "word2idx.pt")
torch.save(idx2word, MODEL_DIR / "idx2word.pt")
print("Saved model and vocab to", MODEL_DIR)

