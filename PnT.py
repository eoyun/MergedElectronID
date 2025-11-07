import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

def plot_pred_vs_true(df, pred_col, true_col):
    """
    Multi-class classifier prediction vs true label comparison table.

    Parameters:
    df (pd.DataFrame): DataFrame containing the prediction and true label columns.
    pred_col (str): Column name for predictions.
    true_col (str): Column name for true labels.
    """
    # Create confusion matrix
    confusion_matrix = pd.crosstab(df[true_col], df[pred_col], rownames=['True'], colnames=['Predicted'])

    # Plot heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt="d", cmap="Blues", cbar=False)
    
    # Titles and labels
    plt.title("Prediction vs True Label")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    plt.savefig("PnT_ENSEMBLE_20251024_2155.png")

df = pd.read_csv("./results/keras/ENSEMBLE_20251024_2155.csv")
print(df)
plot_pred_vs_true(df,"pred","true")

