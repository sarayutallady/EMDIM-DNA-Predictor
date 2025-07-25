import streamlit as st
import torch
import pickle
from utils.preprocessing import encode_sequence
from utils.visualization import plot_mutation_map
from utils.interpretability import interpret_top_kmers
from models.emdim_model import EMDIM
import matplotlib.pyplot as plt

# === CONFIG ===
K = 6
MAX_SEQ_LEN = 256
MODEL_PATH = "models/emdim_model.pt"
VOCAB_PATH = "models/vocab.pkl"

# === LOAD VOCAB + MODEL ===
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

model = EMDIM(len(vocab))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# === PAGE CONFIG ===
st.set_page_config(page_title="EMDIM DNA Predictor", layout="centered")
st.title("🧬 EMDIM: Explainable Multi-Task DNA Intelligence Model")

# === SIDEBAR: HOW TO USE ===
st.sidebar.title("📘 How to Use")

st.sidebar.markdown("""
This tool helps you analyze DNA sequences to:

🔍 Predict if it's **Healthy** or **Diseased**  
📈 Show **Disease Risk Score**  
🧬 Visualize **Mutation Map**  
🧠 See **Explainability via k-mers**

---

### 🧬 Instructions:
1. Paste a DNA sequence (A, C, G, T only).
2. Or click **Load Example DNA Sequence**.
3. Click **🔍 Predict** to view results.

✅ Works best with ~256 base pairs.

---

Made with ❤️ for real DNA explainability.
""")

# === EXAMPLE DNA SEQUENCE ===
default_example = (
    "ACGTAGTCGATCGTAGCTAGCTGATCGATCGATGCTAGCTAGCTAGCTAGCATGCTAGCATGCTAGCTAGCTGATCGATCGATGCTAGCTAGC"
    "TACGTAGTCGATCGTAGCTAGCTGATCGATCGATGCTAGCTAGCTAGCTAGCATGCTAGCATGCTAGCTAGCTGATCGATCGATGCTAGCTAGC"
)

# === INPUT DNA SEQUENCE ===
st.subheader("📥 Paste Your DNA Sequence")

# Session state flag to control example loading
if 'example_loaded' not in st.session_state:
    st.session_state.example_loaded = False

# Load Example Button
if st.button("🧬 Click here to load an example DNA sequence"):
    st.session_state.example_loaded = True

# Display text area with example only if loaded
sequence = st.text_area(
    "Only characters A, C, G, T are allowed (~256 bp recommended)",
    value=default_example if st.session_state.example_loaded else "",
    height=120,
    max_chars=300
).strip().upper()

# === RUN PREDICTION ===
if sequence and st.button("🔍 Predict"):
    # Encode sequence
    encoded = encode_sequence(sequence, vocab, k=K, max_len=MAX_SEQ_LEN)
    input_tensor = torch.tensor([encoded])

    # Model Prediction
    with torch.no_grad():
        class_logits, mutation_map, risk_score = model(input_tensor)
        class_prob = torch.softmax(class_logits, dim=1).squeeze()
        pred_label = torch.argmax(class_prob).item()
        risk = risk_score.item()
        mutation_probs = torch.sigmoid(mutation_map).squeeze().tolist()

    label_str = "Healthy" if pred_label == 0 else "Diseased"
    confidence = class_prob[pred_label].item()

    # === DISPLAY OUTPUT ===
    st.markdown("### 📌 Prediction Result")
    st.markdown(f"#### 📘 Class Label: **{label_str}** (prob: {confidence:.4f})")
    st.markdown(f"#### ⚠️  Disease Risk Score: **{risk:.4f}**")

    # Explainability
    top_k_info, summary = interpret_top_kmers(sequence, mutation_probs, k=K, top_n=3)
    st.markdown(f"### 🧠 Explainability Report for Prediction: **{label_str}**")
    for kmer, pos, score in top_k_info:
        st.write(f"- K-mer '{kmer}' at position {pos}: importance score {score:.4f}")

    st.markdown("**🧠 Explainability Summary:**")
    st.markdown(summary)

    # Mutation Map
    st.markdown("### 🧬 Mutation Map (first 20 positions):")
    st.write(mutation_probs[:20])

    # Plot Graph
    st.markdown("### 📊 Mutation Visualization:")
    fig, ax = plt.subplots(figsize=(16, 2))
    ax.bar(range(len(mutation_probs)), mutation_probs, color='tomato')
    ax.set_title("Mutation Map")
    ax.set_xlabel("Base Position")
    ax.set_ylabel("Mutation Score")
    st.pyplot(fig)

# === FOOTER ===
st.markdown("""---  
Made with ❤️ for real DNA explainability.
""")

