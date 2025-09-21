import pandas as pd
import matplotlib.pyplot as plt

def plot_metrics(csv_path="training_metrics.csv", save_path="metrics_plot.png"):
    df = pd.read_csv(csv_path)

    plt.figure(figsize=(14, 5))

    # Plot Loss
    plt.subplot(1, 3, 1)
    plt.plot(df["epoch"], df["loss"], marker='o')
    plt.title("Loss over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Loss")
    plt.grid(True)

    # Plot Accuracy
    plt.subplot(1, 3, 2)
    plt.plot(df["epoch"], df["accuracy"], marker='o', color='green')
    plt.title("Accuracy over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("Accuracy")
    plt.grid(True)

    # Plot ROC AUC
    plt.subplot(1, 3, 3)
    plt.plot(df["epoch"], df["roc_auc"], marker='o', color='orange')
    plt.title("ROC AUC over Epochs")
    plt.xlabel("Epoch")
    plt.ylabel("ROC AUC")
    plt.grid(True)

    plt.tight_layout()

    # Save plot to file
    plt.savefig(save_path, dpi=300)

    # Show plot
    plt.show()

if __name__ == "__main__":
    plot_metrics()
