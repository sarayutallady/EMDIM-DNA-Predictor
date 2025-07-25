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

# === LOAD MODEL AND VOCAB ===
with open(VOCAB_PATH, "rb") as f:
    vocab = pickle.load(f)

model = EMDIM(len(vocab))
model.load_state_dict(torch.load(MODEL_PATH, map_location="cpu"))
model.eval()

# === PAGE SETTINGS ===
st.set_page_config(page_title="EMDIM DNA Predictor", layout="centered")
st.title("🧬 EMDIM: Explainable Multi-Task DNA Intelligence Model")
st.markdown("""
This app uses an AI model to classify DNA sequences as **Healthy** or **Diseased**, predict **mutation-prone regions**, and explain the results using top **k-mer tokens**.  
Paste your DNA sequence or use the sample to get started.
""")

# === SIDEBAR HELP ===
st.sidebar.title("❓ How to Use")
st.sidebar.markdown("""
1. Paste your DNA sequence (A, T, C, G only)  
2. Or click **Sample Input**  
3. Click **🔍 Predict**  
4. View class label, risk score, mutation map, and explainability
""")

# === SAMPLE DNA BUTTON ===
sample_seq = "AGTCAGTCAGTCAGTTAGCGTAACGTAGCTAGCTAGCTAGTACG"
if st.button("📋 Use Sample DNA Sequence"):
    st.session_state["dna_input"] = sample_seq

# === USER INPUT ===
st.subheader("📥 Paste Your DNA Sequence")
sequence = st.text_area("Only A, C, G, T characters allowed (~256 bp recommended)",
                        value=st.session_state.get("dna_input", "")).strip().upper()

# === PREDICTION ===
if sequence and st.button("🔍 Predict"):
    # Encode sequence
    encoded = encode_sequence(sequence, vocab, k=K, max_len=MAX_SEQ_LEN)
    input_tensor = torch.tensor([encoded])

    # Run Model
    with torch.no_grad():
        class_logits, mutation_map, risk_score = model(input_tensor)
        class_prob = torch.softmax(class_logits, dim=1).squeeze()
        pred_label = torch.argmax(class_prob).item()
        risk = risk_score.item()
        mutation_probs = torch.sigmoid(mutation_map).squeeze().tolist()

    label_str = "Healthy" if pred_label == 0 else "Diseased"
    confidence = class_prob[pred_label].item()

    # === OUTPUT ===
    st.markdown("### 📌 Prediction Result")
    st.markdown(f"#### 📘 Class Label: **{label_str}** (Confidence: {confidence:.4f})")
    st.markdown(f"#### ⚠️ Disease Risk Score: **{risk:.4f}**")

    # Explainability
    top_k_info, summary = interpret_top_kmers(sequence, mutation_probs, k=K, top_n=3)
    st.markdown(f"### 🧠 Explainability Report")
    for kmer, pos, score in top_k_info:
        st.write(f"- K-mer '{kmer}' at position {pos} → importance score: {score:.4f}")
    st.markdown("**🧠 Summary:**")
    st.markdown(summary)

    # Mutation Table
    st.markdown("### 🧬 Mutation Map (First 20 Bases):")
    st.write(mutation_probs[:20])

    # Mutation Visualization
    st.markdown("### 📊 Mutation Map Visualization:")
    fig, ax = plt.subplots(figsize=(16, 2))
    ax.bar(range(len(mutation_probs)), mutation_probs, color='tomato')
    ax.set_title("Mutation Map")
    ax.set_xlabel("Base Position")
    ax.set_ylabel("Mutation Score")
    st.pyplot(fig)

elif not sequence:
    st.info("👈 Paste a sequence or click '📋 Use Sample DNA Sequence' to try the model.")
    st.markdown("### 🔬 Example Output (Preview)")
    st.markdown("- **Prediction:** Diseased (Confidence: 0.89)")
    st.markdown("- **Risk Score:** 0.85")
    st.markdown("- **Top kmers:** GTA, TAC, CAG")
    st.markdown("You can run real predictions once you paste a sequence above!")

# === FOOTER ===
st.markdown("---")
st.markdown("Made with ❤️ for real-world DNA sequence analysis.")

