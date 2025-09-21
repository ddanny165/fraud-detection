import sys
import os
import warnings
import numpy as np
import json
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.model_selection import ParameterSampler
from scipy.stats import uniform

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_data, preprocess_data, split_data

warnings.filterwarnings("ignore", category=RuntimeWarning)

def get_param_distribution():
    return {
        'C': uniform(loc=0.001, scale=10),
        'penalty': ['l1', 'l2'],
        'class_weight': [{0: 1, 1: w} for w in np.linspace(5, 20, 20)],
        'solver': ['liblinear'],
        'max_iter': [1000]
    }

def evaluate_model(model, X_val, y_val):
    y_pred = model.predict(X_val)
    y_proba = model.predict_proba(X_val)[:, 1]
    print("\nValidation Set Evaluation:")
    print(classification_report(y_val, y_pred, digits=4))
    print("ROC AUC Score (Val):", roc_auc_score(y_val, y_proba))
    return roc_auc_score(y_val, y_proba)

def tune_hyperparameters(X_train, y_train, X_val, y_val, param_dist, n_iter=50):
    param_list = list(ParameterSampler(param_dist, n_iter=n_iter, random_state=42))
    best_score = -np.inf
    best_model = None
    best_params = None

    for params in param_list:
        try:
            model = LogisticRegression(**params)
            model.fit(X_train, y_train)
            score = roc_auc_score(y_val, model.predict_proba(X_val)[:, 1])
            if score > best_score:
                best_score = score
                best_model = model
                best_params = params
        except Exception as e:
            print(f"Skipped params {params} due to error: {e}")
            continue

    print("Best Hyperparameters:", best_params)
    return best_model

def save_model_and_scaler(model, scaler, coef_path="model_logreg_weights.json", scaler_path="scaler_logreg.json"):
    # Save only model coefficients
    model_data = {
        "coef_": model.coef_.tolist(),
        "intercept_": model.intercept_.tolist(),
        "classes_": model.classes_.tolist()
    }
    with open(coef_path, "w") as f:
        json.dump(model_data, f)

    # Save scaler
    scaler_data = {
        "mean_": scaler.mean_.tolist(),
        "scale_": scaler.scale_.tolist(),
        "var_": scaler.var_.tolist(),
        "n_features_in_": scaler.n_features_in_
    }
    with open(scaler_path, "w") as f:
        json.dump(scaler_data, f)

    print("Model and scaler saved as JSON.")

def main():
    X, y = load_data()
    X_scaled, scaler = preprocess_data(X)
    X_train, X_val, _, y_train, y_val, _ = split_data(X_scaled, y)

    param_dist = get_param_distribution()
    best_model = tune_hyperparameters(X_train, y_train, X_val, y_val, param_dist)

    evaluate_model(best_model, X_val, y_val)
    save_model_and_scaler(best_model, scaler)

if __name__ == "__main__":
    main()
