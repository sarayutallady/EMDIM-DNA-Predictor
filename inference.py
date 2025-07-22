import torch
import pickle
from models.emdim_model import EMDIM
from utils.preprocessing import encode_sequence, kmer_encode
from utils.visualization import plot_mutation_map
from utils.interpretability import interpret_top_kmers

# === CONFIG ===
K = 6
MAX_SEQ_LEN = 256
MODEL_PATH = "models/emdim_model.pt"
VOCAB_PATH = "models/vocab.pkl"

# === LOAD VOCAB ===
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

# === GET INPUT FROM USER ===
sequence = input("🧬Enter your DNA sequence (A, C, G, T only):\n").strip().upper()
seq_id = "user_input"

# === ENCODE INPUT ===
encoded = encode_sequence(sequence, vocab, k=K, max_len=MAX_SEQ_LEN)
input_tensor = torch.tensor([encoded])

# === LOAD MODEL ===
model = EMDIM(len(vocab))
model.load_state_dict(torch.load(MODEL_PATH, map_location=torch.device("cpu")))
model.eval()

# === PREDICT ===
with torch.no_grad():
    class_logits, mutation_map, risk_score = model(input_tensor)
    class_probs = torch.softmax(class_logits, dim=1).squeeze()
    class_label = torch.argmax(class_probs).item()
    risk = risk_score.item()
    mutation_probs = torch.sigmoid(mutation_map).squeeze().tolist()

# === CLASS LABEL TEXT ===
label_text = "Diseased" if class_label == 1 else "Healthy"
label_prob = class_probs[class_label].item()

# === PRINT BASIC OUTPUT ===
print(f"\n[📌 {seq_id}]")
#print(f"🦜 Sequence: {sequence[:60]}.. ")
print(f"📘 Class Label: {label_text} (prob: {label_prob:.4f})")
print(f"⚠️  Disease Risk Score: {risk:.4f}")

# === INTERPRETABILITY OUTPUT ===
top_k_info, summary = interpret_top_kmers(sequence, mutation_map, k=K, top_n=3)

print(f"\n🧠 Explainability Report for Prediction: {label_text}")
for pos, kmer, score in top_k_info:
    print(f"  - K-mer '{kmer}' at position {pos}: importance score {score:.4f}")
print(f"\n🧠 Explainability Summary:\n{summary}")

# === MUTATION MAP ===
print("\n🧬 Mutation Map (first 20 positions):")
print(mutation_probs[:20])

# === VISUALIZATION ===
plot_mutation_map(sequence, mutation_probs, title=f"Mutation Map: {seq_id}")
