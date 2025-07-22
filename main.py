import pandas as pd
import torch
import torch.nn as nn
import torch.optim as optim
import pickle
from sklearn.model_selection import train_test_split
from torch.utils.data import Dataset, DataLoader
from tqdm import tqdm
from utils.preprocessing import build_vocab, encode_sequence
from models.emdim_model import EMDIM

# === CONFIG ===
K = 6
MAX_SEQ_LEN = 256
BATCH_SIZE = 64
EPOCHS = 5
MODEL_PATH = "models/emdim_model.pt"
VOCAB_PATH = "models/vocab.pkl"
DATA_PATH = "data/clinvar_dataset_20k.csv"

# === LOAD DATA ===
print("📥 Loading dataset...")
df = pd.read_csv(DATA_PATH)

# Ensure correct label format
df["label"] = df["label"].astype(int)

# === BUILD VOCAB ===
print("🧠 Building k-mer vocabulary...")
vocab = build_vocab(df["sequence"].values, k=K)
vocab_size = len(vocab)
print(f"✅ Vocab size: {vocab_size}")

# === DATASET CLASS ===
class DNADataset(Dataset):
    def __init__(self, df, vocab):
        self.df = df
        self.vocab = vocab

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        row = self.df.iloc[idx]
        x = encode_sequence(row["sequence"], self.vocab, K, MAX_SEQ_LEN)
        y_class = row["label"]
        y_risk = float(y_class)  # Disease risk score = same as class
        return torch.tensor(x), torch.tensor(y_class), torch.tensor(y_risk)

# === SPLIT DATA ===
train_df, _ = train_test_split(df, test_size=0.2, random_state=42)
train_ds = DNADataset(train_df, vocab)
train_loader = DataLoader(train_ds, batch_size=BATCH_SIZE, shuffle=True)

# === MODEL ===
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
model = EMDIM(vocab_size).to(device)
optimizer = optim.Adam(model.parameters(), lr=1e-3)
criterion_class = nn.CrossEntropyLoss()
criterion_risk = nn.MSELoss()

# === TRAINING ===
print("🚀 Starting training...\n")
for epoch in range(EPOCHS):
    model.train()
    total_loss, correct, total = 0, 0, 0

    for x, y_class, y_risk in tqdm(train_loader, desc=f"Epoch {epoch+1}/{EPOCHS}"):
        x = x.to(device)
        y_class = y_class.to(device)
        y_risk = y_risk.to(device)

        optimizer.zero_grad()
        class_logits, mutation_map, risk_score = model(x)

        loss_class = criterion_class(class_logits, y_class)
        loss_risk = criterion_risk(risk_score.squeeze(), y_risk)
        loss = loss_class + loss_risk  # total loss

        loss.backward()
        optimizer.step()

        total_loss += loss.item()
        preds = class_logits.argmax(dim=1)
        correct += (preds == y_class).sum().item()
        total += y_class.size(0)

    acc = correct / total * 100
    print(f"✅ Epoch {epoch+1} | Loss: {total_loss:.4f} | Accuracy: {acc:.2f}%")

# === SAVE MODEL + VOCAB ===
torch.save(model.state_dict(), MODEL_PATH)
with open(VOCAB_PATH, "wb") as f:
    pickle.dump(vocab, f)

print(f"\n💾 Model saved as {MODEL_PATH}")
print(f"🧠 Vocab saved as {VOCAB_PATH}")
