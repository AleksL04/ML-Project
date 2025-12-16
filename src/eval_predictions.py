import pandas as pd
from sklearn.metrics import classification_report

df = pd.read_csv('predictions.csv')

TRUE_LABEL_COL = 'label'
PRED_LABEL_COL = 'prediction'

if TRUE_LABEL_COL not in df.columns:
    print(f"Error: Column '{TRUE_LABEL_COL}' not found in CSV.")
    print("Available columns:", df.columns.tolist())
else:
    y_true = df[TRUE_LABEL_COL]
    y_pred = df[PRED_LABEL_COL]

    print("Performance on Open Test Set")
    print(classification_report(y_true, y_pred, digits=4))