# 🛡️ Fraud Detection using Machine Learning

This project implements and evaluates machine learning models for **fraud detection in financial transactions**.  
The goal is to classify each transaction as **fraudulent (1)** or **legitimate (0)**, using an **imbalanced dataset** of ~228k transactions.

---

## 📊 Dataset

- **Total samples**: 227,845
- **Fraud cases**: 394 (0.173%) → _highly imbalanced_
- **Features**: 28 anonymized signals + raw features (`Time`, `Amount`)
- **Target**: Binary (Fraud / Legitimate)

⚠️ **Note**: This repository only contains a **sample dataset** (`transactions.csv`, ~1MB) for demonstration and testing purposes.  
The full dataset (>100 MB) is not included due to GitHub limits.

If you want to train models properly with a full dataset, please contact me directly.

---

## ⚙️ Implementation

### 🔹 Data Preprocessing

- Removed low-variance features
- Standardized data using `StandardScaler`
- Train/validation/test split (75% / 20% / 5%)

---

## 📐 Hyperparameter Tuning

Hyperparameters were optimized using **randomized search** with validation ROC AUC as the objective metric.  
This approach helps avoid bias from the imbalanced dataset (accuracy alone would be misleading).

### Logistic Regression

- Search space: `C` (inverse regularization), penalty type (`l1`, `l2`), fraud class weight
- Method: 50 random samples using scikit-learn’s `ParameterSampler`
- Final choice:
  - `C = 0.1336`
  - `penalty = l2`
  - `fraud class weight = 15.263`

### Multilayer Perceptron (MLP)

- Search space: hidden size, batch size, learning rate, epochs, fraud class weight
- Method: 60 random trials with mini-batch training
- Final choice:
  - hidden size = 32
  - batch size = 32
  - learning rate = 0.000736
  - epochs = 5
  - fraud class weight = 3.544

📌 Using ROC AUC instead of accuracy ensured that the models truly improved at fraud detection rather than just predicting the majority (legitimate) class.

---

### 🔹 Handling Class Imbalance

- Applied **class weighting** (higher penalty for fraud misclassification)
- Considered but did not use oversampling due to overfitting

---

### 🔹 Models

1. **Logistic Regression** (baseline)

   - Implemented with scikit-learn
   - Optimized using cross-entropy loss
   - Tuned `C`, regularization type, and class weight

2. **Multilayer Perceptron (MLP)**
   - Implemented in PyTorch
   - Architecture: 2 hidden layers, ReLU activations, sigmoid output
   - Loss: `BCEWithLogitsLoss` with fraud class weighting
   - Optimizer: Adam
   - Tuned hidden size, batch size, learning rate, epochs, and class weight

---

## 🔍 Results

| Model                | ROC AUC (Validation) | Fraud Precision | Fraud Recall | Fraud F1 |
| -------------------- | -------------------- | --------------- | ------------ | -------- |
| Logistic Regression  | 0.9636               | 0.71            | 0.76         | 0.74     |
| MLP (Neural Network) | **0.9956**           | **0.86**        | **0.86**     | **0.86** |

- **Test Set (MLP)**: ROC AUC = **0.9929**, Precision = 0.94, Recall = 0.85
- Logistic Regression performed well as a baseline, but **MLP consistently outperformed it**.

---

## 📈 Key Takeaways

- **Class imbalance handling is essential** (class weights improved recall drastically).
- **ROC AUC** is more meaningful than accuracy for imbalanced datasets.
- Logistic Regression is a **strong, interpretable baseline**.
- Neural Networks (MLP) can **capture more complex patterns** and boost performance further.

---

## 📌 Future Work

- Experiment with **deeper neural networks**
- Apply **threshold tuning** for precision-recall balance
- Explore **alternative imbalance handling techniques** (SMOTE, ensemble methods)

---

## 🧑‍💻 Author

**Danylo Moskaliuk**  
University of Innsbruck – Machine Learning (SS 2025)
