import pandas as pd
from sklearn.metrics import confusion_matrix

def main():
    # Load data
    df = pd.read_csv('predictions.csv')
    
    # Check for correct column names
    if 'label' not in df.columns or 'prediction' not in df.columns:
        print("Error: CSV must contain 'label' and 'prediction' columns.")
        return

    y_true = df['label']
    y_pred = df['prediction']

    # Calculate matrix
    cm = confusion_matrix(y_true, y_pred)
    tn, fp, fn, tp = cm.ravel()

    # Print clean text output
    print("Confusion Matrix")
    print(f"                 Predicted: 0   Predicted: 1")
    print(f"Actual: 0       {tn:<14} {fp}")
    print(f"Actual: 1       {fn:<14} {tp}")

if __name__ == "__main__":
    main()