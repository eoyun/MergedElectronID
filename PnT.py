import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
import glob
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay
import numpy as np
import joblib
from sklearn.preprocessing import label_binarize


def plot_roc_curve(y_true, y_pred, labels, num_classes):
    y_true_bin = label_binarize(y_true, classes=labels) 
    y_pred_bin = label_binarize(y_pred, classes=labels) 
    plt.figure(figsize=(10, 10))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_bin[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {labels[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")  # L~@AD|  기@D|
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"./roc_test.png")


def plot_pred_vs_true(df, pred_col, true_col):
    """
    Multi-class classifier prediction vs true label comparison table.

    Parameters:
    df (pd.DataFrame): DataFrame containing the prediction and true label columns.
    pred_col (str): Column name for predictions.
    true_col (str): Column name for true labels.
    """
    # Create confusion matrix
    confusion_matrix = pd.crosstab(df[true_col], df[pred_col],normalize = 'index', rownames=['True'], colnames=['Predicted'])

    # Plot heatmap
    plt.figure(figsize=(5, 4))
    sns.heatmap(confusion_matrix, annot=True, fmt="f", cmap="Blues", cbar=False)
    
    # Titles and labels
    plt.title("Prediction vs True Label")
    plt.xlabel("Predicted Label")
    plt.ylabel("True Label")
    
    plt.savefig("PnT_wo_add_trk_20251124_1726.png")

files = glob.glob("./results/keras/wo_add_trk_20251124_1726_df_*.csv")
dfs = [pd.read_csv(f) for f in files]

df = pd.concat(dfs,ignore_index=True)

#df = pd.read_csv("./results/keras/wo_add_trk_20251120_1916.csv")
print(df)
ohe = joblib.load("ohe.pkl")
labels = np.unique(df["Label"])
plot_pred_vs_true(df,"prediction","Label")
y_pred = df['prediction'].to_numpy()
y_true = df['Label'].to_numpy()
ohe.transform(y_pred.reshape(-1,1))
ohe.transform(y_true.reshape(-1,1))

plot_roc_curve( y_true,y_pred, labels, 3)


