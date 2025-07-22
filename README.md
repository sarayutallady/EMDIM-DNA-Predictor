# 🧬 EMDIM: Explainable Multi-Task DNA Intelligence Model

**EMDIM-DNA-Predictor** is an explainable, multi-task deep learning model built to analyze real human DNA sequences for intelligent disease prediction.

It performs:
- ✅ **DNA Classification** (Healthy / Diseased)
- 🔍 **Mutation Prediction** (mutation-prone base positions)
- ⚠️ **Disease Risk Scoring**
- 🧠 **Explainability** with k-mer importance
- 📊 **Visualization** of mutation heatmaps

---

## 📌 Key Features

✅ Real DNA Data (ClinVar + Human Genome)  
✅ Multi-Task Learning: Classification, Mutation Mapping, Risk Scoring  
✅ Interpretability: Top influential k-mers with scores and positions  
✅ Streamlit Web App for interaction  
✅ Downloadable Reports and Visual Graphs

---

## 📊 Outputs

For any given DNA input, the model outputs:

- **📘 Class Label**: Healthy / Diseased
- **⚠️ Disease Risk Score**: Range from 0.0 (healthy) to 1.0 (high risk)
- **🧠 Explainability Report**: Top 3 k-mers with position & score
- **📊 Mutation Heatmap**: Bar graph of high-risk mutation areas
- **📥 Download Options**: Text & Image outputs

---

## 🔬 How It Works

1. **Preprocessing**:
   - Splits DNA into overlapping 6-mers (k=6)
   - Encodes them using a learned vocabulary

2. **Model Prediction**:
   - Transformer model processes sequence embeddings
   - Outputs:
     - Healthy/Diseased classification
     - Mutation map with importance scores
     - Disease risk score

3. **Explainability**:
   - Uses importance scores to extract top k-mers
   - Provides natural-language summary of key contributors

4. **Visualization**:
   - Bar graph showing base-level mutation likelihoods
   - Printable and downloadable

---

## 📈 Performance Metrics

| Metric                  | Score   |
|-------------------------|---------|
| **Accuracy**            | ~0.96   |
| **F1 Score**            | ~0.96   |
| **Mean Absolute Error** | ~0.038  |
| **R² Score**            | ~0.97   |

---

