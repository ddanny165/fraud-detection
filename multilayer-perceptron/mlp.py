import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.utils.data import DataLoader, TensorDataset
import sys
import os
import argparse
import json
import numpy as np
import random
import pandas as pd
from sklearn.metrics import classification_report, roc_auc_score, accuracy_score

# Random seed for reproducibility
random_seed = 123
torch.manual_seed(random_seed)
np.random.seed(random_seed)
random.seed(random_seed)

# Default fixed hyperparameters obtained via search 
DEFAULT_PARAMS = {
    'hidden_size': 32,
    'batch_size': 32,
    'learning_rate': 0.000736,
    'fraud_class_weight': 3.544,
    'epochs': 5
}

sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils import load_data, preprocess_data, split_data

class MLP(nn.Module):
    def __init__(self, input_size, hidden_size, output_size):
        super(MLP, self).__init__()
        self.fc1 = nn.Linear(input_size, hidden_size)
        self.fc2 = nn.Linear(hidden_size, hidden_size)
        self.fc3 = nn.Linear(hidden_size, output_size)

    def forward(self, x):
        x = F.relu(self.fc1(x))
        x = F.relu(self.fc2(x))
        return self.fc3(x)

def evaluate_model(model, X, y, name="Dataset"):
    model.eval()
    with torch.no_grad():
        inputs = torch.tensor(X, dtype=torch.float32)
        logits = model(inputs).squeeze()
        probs = torch.sigmoid(logits).numpy()
        preds = (probs >= 0.5).astype(int)
        print(f"{name} Evaluation:")
        print(classification_report(y, preds))
        print(f"ROC AUC Score ({name}):", roc_auc_score(y, probs))

def train_model(model, train_loader, criterion, optimizer, epochs, save_log=True):
    model.train()
    training_history = []

    for epoch in range(epochs):
        epoch_loss = 0
        all_labels = []
        all_probs = []

        for batch_X, batch_y in train_loader:
            optimizer.zero_grad()
            logits = model(batch_X).squeeze()
            loss = criterion(logits, batch_y)
            loss.backward()
            optimizer.step()
            epoch_loss += loss.item()

            with torch.no_grad():
                probs = torch.sigmoid(logits)
                all_labels.extend(batch_y.numpy())
                all_probs.extend(probs.numpy())

        preds_binary = (np.array(all_probs) >= 0.5).astype(int)
        acc = accuracy_score(all_labels, preds_binary)
        roc = roc_auc_score(all_labels, all_probs)
        loss_avg = epoch_loss / len(train_loader)

        print(f"Epoch {epoch + 1}/{epochs} | Loss: {loss_avg:.4f} | Accuracy: {acc:.4f} | ROC AUC: {roc:.4f}")

        training_history.append({
            "epoch": epoch + 1,
            "loss": loss_avg,
            "accuracy": acc,
            "roc_auc": roc
        })

    if save_log:
        pd.DataFrame(training_history).to_csv("training_metrics.csv", index=False)
        print("Training metrics saved to training_metrics.csv")

def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--train', action='store_true', help='Train the model with fixed hyperparameters')
    parser.add_argument('--search', action='store_true', help='Search for best hyperparameters')
    args = parser.parse_args()

    X, y = load_data()
    X_scaled, scaler = preprocess_data(X)
    scaler_params = {
        "mean": scaler.mean_.tolist(),
        "scale": scaler.scale_.tolist()
    }
    X_train, X_val, X_test, y_train, y_val, y_test = split_data(X_scaled, y)
    input_size, output_size = X_train.shape[1], 1

    if args.search:
        print("Searching for best hyperparameters...")
        best_auc, best_params = 0, {}

        for trial in range(60):
            params = {
                'hidden_size': random.choice([8, 16, 32, 64, 128]),
                'batch_size': random.choice([16, 32, 64]),
                'learning_rate': random.uniform(1e-4, 1e-2),
                'fraud_class_weight': random.uniform(1.5, 5.0),
                'epochs': random.choice([5, 10, 20, 30])
            }

            model = MLP(input_size, params['hidden_size'], output_size)
            train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
            train_loader = DataLoader(train_dataset, batch_size=params['batch_size'], shuffle=True)

            criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([params['fraud_class_weight']]))
            optimizer = torch.optim.Adam(model.parameters(), lr=params['learning_rate'])

            train_model(model, train_loader, criterion, optimizer, params['epochs'], save_log=False)

            with torch.no_grad():
                val_outputs = model(torch.tensor(X_val, dtype=torch.float32)).squeeze()
                val_probs = torch.sigmoid(val_outputs).numpy()
                auc = roc_auc_score(y_val, val_probs)

            if auc > best_auc:
                best_auc = auc
                best_params = params

        print("Best hyperparameters found:", best_params)
        model = MLP(input_size, best_params['hidden_size'], output_size)
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=best_params['batch_size'], shuffle=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([best_params['fraud_class_weight']]))
        optimizer = torch.optim.Adam(model.parameters(), lr=best_params['learning_rate'])
        train_model(model, train_loader, criterion, optimizer, best_params['epochs'])
        torch.save(model.state_dict(), "mlp_model.pth")
        with open("scaler_params.json", "w") as f:
            json.dump(scaler_params, f)
        print("Model and scaler saved.")
        evaluate_model(model, X_val, y_val, name="Validation Set")

    elif args.train:
        print("Training with predefined hyperparameters...")
        model = MLP(input_size, DEFAULT_PARAMS['hidden_size'], output_size)
        train_dataset = TensorDataset(torch.tensor(X_train, dtype=torch.float32), torch.tensor(y_train.values, dtype=torch.float32))
        train_loader = DataLoader(train_dataset, batch_size=DEFAULT_PARAMS['batch_size'], shuffle=True)
        criterion = nn.BCEWithLogitsLoss(pos_weight=torch.tensor([DEFAULT_PARAMS['fraud_class_weight']]))
        optimizer = torch.optim.Adam(model.parameters(), lr=DEFAULT_PARAMS['learning_rate'])
        train_model(model, train_loader, criterion, optimizer, DEFAULT_PARAMS['epochs'])
        torch.save(model.state_dict(), "mlp_model.pth")
        with open("scaler_params.json", "w") as f:
            json.dump(scaler_params, f)
        print("Model and scaler saved.")
        evaluate_model(model, X_val, y_val, name="Validation Set")

    else:
        print("Loading saved model...")
        model = MLP(input_size, DEFAULT_PARAMS['hidden_size'], output_size)
        model.load_state_dict(torch.load("mlp_model.pth"))
        model.eval()
        evaluate_model(model, X_train, y_train, name="Train Set")
        evaluate_model(model, X_test, y_test, name="Test Set")

if __name__ == "__main__":
    main()