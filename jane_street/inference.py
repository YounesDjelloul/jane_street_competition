import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import pandas as pd
import polars as pl
from sklearn.model_selection import train_test_split
import os
import kaggle_evaluation.jane_street_inference_server


model_path = '/kaggle/input/iteration-0/pytorch/default/1/best_model.pth'

model = None
feature_cols = None
lags_ = None
current_lag_id = 0

def initialize_model(input_size):
    model = FinancialNN(input_size)
    model.load_state_dict(torch.load(model_path))
    model.eval()
    return model


def prepare_features(df, current_lag_id, lags_df=None):
    columns_to_drop = ['feature_21', 'feature_26', 'feature_27', 'feature_31']

    if isinstance(df, pl.DataFrame):
        df = df.to_pandas()

    feature_cols = [f'feature_{i:02d}' for i in range(79) if f'feature_{i:02d}' not in columns_to_drop]

    feature_cols = [col for col in feature_cols if col in df.columns]

    responder_cols = [f'responder_{i}_lag_{current_lag_id}' for i in range(9) if i != 6]

    features_df = df.copy()

    if lags_df is not None:
        try:
            if isinstance(lags_df, pl.DataFrame):
                lags_df = lags_df.to_pandas()

            available_responders = [col for col in responder_cols if col in lags_df.columns]
            for col in available_responders:
                features_df[col] = lags_df[col].values

            missing_responders = [col for col in responder_cols if col not in available_responders]
            for col in missing_responders:
                features_df[col] = 0.0
        except Exception as e:
            print(f"Warning: Error processing lags data: {str(e)}")
            for col in responder_cols:
                features_df[col] = 0.0
    else:
        for col in responder_cols:
            features_df[col] = 0.0

    all_features = feature_cols + responder_cols

    for col in all_features:
        median_val = features_df[col].median()

        if features_df[col].isna().any():
            features_df[col] = features_df[col].fillna(median_val if not pd.isna(median_val) else 0)

    return features_df[all_features].values


def predict(test: pl.DataFrame, lags: pl.DataFrame | None) -> pl.DataFrame:
    global model, lags_, current_lag_id

    if lags is not None:
        lags_ = lags
        current_lag_id += 1

    if model is None:
        input_features = 83
        model = initialize_model(input_features)
        print("Model initialized")

    try:
        features = prepare_features(test, current_lag_id, lags_ if lags is None else lags)
        print(f"Feature shape during training: {features}")

        with torch.no_grad():
            features_tensor = torch.FloatTensor(features)
            predictions = model(features_tensor).numpy().flatten()

        predictions_df = pl.DataFrame({
            'row_id': test['row_id'],
            'responder_6': predictions
        })
        return predictions_df

    except Exception as e:
        print(f"Error during prediction: {str(e)}")
        return pl.DataFrame({
            'row_id': test['row_id'],
            'responder_6': np.zeros(len(test))
        })

inference_server = kaggle_evaluation.jane_street_inference_server.JSInferenceServer(predict)

if os.getenv('KAGGLE_IS_COMPETITION_RERUN'):
    inference_server.serve()
else:
    inference_server.run_local_gateway(
        (
            '/kaggle/input/jane-street-real-time-market-data-forecasting/test.parquet',
            '/kaggle/input/jane-street-real-time-market-data-forecasting/lags.parquet',
        )
    )