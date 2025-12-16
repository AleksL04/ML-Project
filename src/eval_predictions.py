import pandas as pd
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score, classification_report

# 1. Load the results
df = pd.read_csv('predictions.csv')

# --- CONFIGURATION ---
# Check your CSV to see what the "Truth" column is named.
# Common names: 'label', 'target', 'is_sarcastic'
TRUE_LABEL_COL = 'label'  # <--- CHANGE THIS if your column has a different name
PRED_LABEL_COL = 'prediction'
# ---------------------

# 2. Check if the truth column exists
if TRUE_LABEL_COL not in df.columns:
    print(f"Error: Column '{TRUE_LABEL_COL}' not found in CSV.")
    print("Available columns:", df.columns.tolist())
else:
    y_true = df[TRUE_LABEL_COL]
    y_pred = df[PRED_LABEL_COL]

    # 3. Calculate Metrics
    acc = accuracy_score(y_true, y_pred)
    prec = precision_score(y_true, y_pred)
    rec = recall_score(y_true, y_pred)
    f1 = f1_score(y_true, y_pred)

    print("=== Performance on Open Test Set ===")
    print(f"Accuracy:  {acc:.4f}")
    print(f"Precision: {prec:.4f}")
    print(f"Recall:    {rec:.4f}")
    print(f"F1 Score:  {f1:.4f}")
    
    print("\n--- Detailed Report ---")
    print(classification_report(y_true, y_pred, digits=4))