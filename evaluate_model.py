from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, confusion_matrix
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
import numpy as np
import pandas as pd

# === Simulated Binary Classification Data (for 20,000 DNA Sequences) ===
# True 0 → Pred 0: 9500 (TN)
# True 0 → Pred 1: 500  (FP)
# True 1 → Pred 0: 375  (FN)
# True 1 → Pred 1: 9625 (TP)

y_true = [0]*9500 + [0]*500 + [1]*375 + [1]*9625
y_pred = [0]*9500 + [1]*500 + [0]*375 + [1]*9625

# === Classification Metrics ===
accuracy = accuracy_score(y_true, y_pred)
precision = precision_score(y_true, y_pred)
recall = recall_score(y_true, y_pred)
f1 = f1_score(y_true, y_pred)

print("\n" + "="*50)
print("         Classification Metrics")
print("="*50)
print(f"Accuracy  : {accuracy:.4f}")
print(f"Precision : {precision:.4f}")
print(f"Recall    : {recall:.4f}")
print(f"F1 Score  : {f1:.4f}")
print("="*50 + "\n")

# === Simulated Regression Metrics ===
y_reg_true = np.random.rand(1000)
y_reg_pred = y_reg_true + np.random.normal(0, 0.1, 1000)

mae = mean_absolute_error(y_reg_true, y_reg_pred)
mse = mean_squared_error(y_reg_true, y_reg_pred)
r2 = r2_score(y_reg_true, y_reg_pred)

print("Risk Score Regression Metrics:")
print(f"MAE : {mae:.16f}")
print(f"MSE : {mse:.16f}")
print(f"R²  : {r2:.16f}")
print("="*50 + "\n")

# === Confusion Matrix ===
conf_matrix = confusion_matrix(y_true, y_pred)

print("Confusion Matrix:")
df_cm = pd.DataFrame(conf_matrix, index=["True 0", "True 1"], columns=["Pred 0", "Pred 1"])
print(df_cm)
print()

# Confusion Matrix Explanation Table
explanation = pd.DataFrame({
    "Predicted 0": ["TN / Correct Healthy", "Diseased predicted as Healthy"],
    "Predicted 1": ["Misclassified Healthy as Diseased", "TP (Correct Diseased)"]
}, index=["True 0", "True 1"])

print("Confusion Matrix Explanation Table:")
print(explanation.to_string())
