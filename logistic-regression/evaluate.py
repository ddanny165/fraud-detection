import sys
import os
import warnings
import json
import numpy as np
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import classification_report, roc_auc_score
from sklearn.preprocessing import StandardScaler

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_data, split_data, log_and_filter_features

warnings.filterwarnings("ignore", category=RuntimeWarning)


def load_scaler_from_json(path="scaler_logreg.json"):
    with open(path, "r") as f:
        scaler_data = json.load(f)

    scaler = StandardScaler()
    scaler.mean_ = np.array(scaler_data["mean_"])
    scaler.scale_ = np.array(scaler_data["scale_"])
    scaler.var_ = np.array(scaler_data["var_"])
    scaler.n_features_in_ = scaler_data["n_features_in_"]
    return scaler


def load_model_from_json(path="model_logreg_weights.json"):
    with open(path, "r") as f:
        model_data = json.load(f)

    model = LogisticRegression()
    model.coef_ = np.array(model_data["coef_"])
    model.intercept_ = np.array(model_data["intercept_"])
    model.classes_ = np.array(model_data["classes_"])
    return model


def evaluate_model(X_scaled, y, model, name="Dataset"):
    y_pred = model.predict(X_scaled)
    y_proba = model.predict_proba(X_scaled)[:, 1]

    print(f"\n{name} Evaluation:")
    print(classification_report(y, y_pred, digits=4))
    print(f"ROC AUC Score ({name}):", roc_auc_score(y, y_proba))


def main():
    # Load and preprocess data
    X_raw, y = load_data()
    X_clean = log_and_filter_features(X_raw)

    # Load and apply saved scaler
    scaler = load_scaler_from_json("scaler_logreg.json")
    X_scaled = scaler.transform(X_clean)

    # Split the data
    X_train, _, X_test, y_train, _, y_test = split_data(X_scaled, y)

    # Load model from JSON
    model = load_model_from_json("model_logreg_weights.json")

    # Evaluate
    evaluate_model(X_train, y_train, model, "Train Set")
    evaluate_model(X_test, y_test, model, "Test Set")


if __name__ == "__main__":
    main()