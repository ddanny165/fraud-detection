import pandas as pd
import numpy as np
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split

def load_data(path="../data/transactions.csv"):
    df = pd.read_csv(path)
    X = df.drop(columns=["Class"])
    y = df["Class"]
    return X, y

def preprocess_data(X):
    # Standard Deviation Filtering - Remove near-constant or zero-variance features
    stds = X.std()
    X = X.loc[:, stds > 1e-6]  # keep only features with some variance

    # Standard scaling
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(X)

    return X_scaled, scaler

def split_data(X, y, val_size=0.2, test_size=0.05, random_state=42):
    X_temp, X_test, y_temp, y_test = train_test_split(
        X, y, test_size=test_size, stratify=y, random_state=random_state
    )
    val_ratio = val_size / (1 - test_size)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=val_ratio, stratify=y_temp, random_state=random_state
    )
    return X_train, X_val, X_test, y_train, y_val, y_test

def log_and_filter_features(X):
    # if "Amount" in X.columns:
    #     X["Amount_log"] = np.log1p(X["Amount"])
    #     X.drop(columns=["Amount"], inplace=True)
    # if "Time" in X.columns:
    #     X["Time_log"] = np.log1p(X["Time"])
    #     X.drop(columns=["Time"], inplace=True)
    stds = X.std()
    return X.loc[:, stds > 1e-6]