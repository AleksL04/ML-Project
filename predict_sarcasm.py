import argparse
import pandas as pd
from tensorflow.keras.models import load_model
from xgboost import XGBClassifier
import numpy as np
from process_df import process_df


def main():
    parser = argparse.ArgumentParser(description="Run sarcasm prediction on a CSV file.")

    parser.add_argument('--input', type=str, required=True, help='Path to the input CSV file')
    parser.add_argument('--output', type=str, required=True, help='Path where the output CSV will be saved')

    args = parser.parse_args()

    input_path = args.input
    output_path = args.output

    LSTM_model = load_model('saved_models/lstm_model.keras')
    CNN_model = load_model('saved_models/cnn_model.keras')
    blender = XGBClassifier() 
    blender.load_model('saved_models/blender_model.json')

    print(f"Loading data from: {input_path}")
    
    input_df = pd.read_csv(input_path)

    X_test = process_df(input_df)

    test_pred_LSTM = LSTM_model.predict(X_test).flatten()
    test_pred_CNN = CNN_model.predict(X_test).flatten()

    X_meta_test = np.column_stack([
        test_pred_LSTM,
        test_pred_CNN
    ])

    predictions = blender.predict(X_meta_test)
    input_df['prediction'] = predictions
    input_df.to_csv(output_path, index=False)
    print(f"Predictions saved to: {output_path}")

if __name__ == "__main__":
    main()