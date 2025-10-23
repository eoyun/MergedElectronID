import numpy as np
import pandas as pd
import tensorflow as tf
import keras.backend as K
from tensorflow import keras
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold
from sklearn.preprocessing import OneHotEncoder
from IPython.display import clear_output
from tqdm import tqdm
from glob import glob

import datetime

def generate_output_filename():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    filename = f"{timestamp}"
    return filename

# 하이퍼파라미터
BATCH_SIZE = 8
EPOCHS = 100
NUM_CLASSES = 3
NUM_META_INPUTS = 4
SEED = 42
datetime = generate_output_filename()
VERSION = f'only_meta_{datetime}'
AUTOTUNE = tf.data.AUTOTUNE

# 메타 + 라벨 불러오기 (기존 Swin 코드와 동일하게 전처리되어 있다고 가정)
df = pd.read_csv("rm_invalid.csv")
df = df.sample(frac=1, random_state=42).reset_index(drop=True)
df['dPhi'] = df['dPhi'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
df['dEta'] = df['dEta'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
df['dEtadPhi'] = df.apply(lambda row: np.concatenate((row['dEta'], row['dPhi'])), axis=1)

# 클래스당 2만개 샘플 제한
df_mergedHard = df[df['Label'] == 'mergedHard'].iloc[:20000]
df_notMerged = df[df['Label'] == 'notMerged'].iloc[:20000]
df_notElectron = df[df['Label'] == 'notElectron'].iloc[:20000]
df_train = pd.concat([df_mergedHard, df_notMerged, df_notElectron], ignore_index=True)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

# 입력과 라벨 분리
meta = np.stack(df_train['dEtadPhi'].to_numpy()).astype(np.float32)
y = df_train['Label'].to_numpy()
labels = np.unique(y)

# 라벨 one-hot 인코딩
ohe = OneHotEncoder(sparse_output=False)
ohe.fit(y.reshape(-1, 1))

# train / valid / test split
#X_train, X_test, y_train, y_test = train_test_split(meta, y_ohe, test_size=0.2, random_state=42)
#X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size=0.5, random_state=42)

# Dataset 구성

class MetricsLogger(keras.callbacks.Callback):
    def __init__(self):
        self.history = {
            "loss": [],
            "val_loss": [],
            "f1": [],
            "val_f1": []
        }
        self.y_true = []
        self.y_pred = []

    def on_epoch_end(self, epoch, logs=None):
        self.history["loss"].append(logs.get("loss"))
        self.history["val_loss"].append(logs.get("val_loss"))
        self.history["f1"].append(logs.get("f1"))
        self.history["val_f1"].append(logs.get("val_f1"))



# 메타 전용 모델
def build_meta_only_model():
    meta_inputs = keras.Input((NUM_META_INPUTS,), name="meta_input")
    x = keras.layers.Dense(32, activation="relu")(meta_inputs)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    model = keras.Model(inputs=meta_inputs, outputs=outputs)
    return model

class DisplayCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            clear_output(wait=True)
metrics_logger = MetricsLogger()
X_tmp, X_test, y_tmp, y_test = train_test_split(meta, y, test_size =0.2)
skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
for fold, (train_idx, valid_idx) in enumerate(skf.split(X_tmp, y_tmp)):
    X_train, y_train = meta[train_idx], y[train_idx]
    X_valid, y_valid = meta[valid_idx], y[valid_idx]
    y_train = ohe.transform(y_train.reshape(-1, 1))
    y_valid = ohe.transform(y_valid.reshape(-1, 1))

    ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    ds_valid = tf.data.Dataset.from_tensor_slices((X_valid, y_valid)).batch(BATCH_SIZE).prefetch(AUTOTUNE)
    
    callbacks = [
        DisplayCallback(),
        keras.callbacks.TensorBoard(log_dir=f"./logs/keras/{VERSION}/fold_{fold}"),
        keras.callbacks.EarlyStopping(monitor="val_f1", mode="max", verbose=0, patience=5),
        keras.callbacks.ModelCheckpoint(f"./ckpts/keras/{VERSION}/fold_{fold}.keras", monitor="val_f1", mode="max", save_best_only=True),
        keras.callbacks.ReduceLROnPlateau(monitor="val_f1", mode="min", factor=0.8, patience=3),
        metrics_logger
    ]
    
    # 모델 학습
    model = build_meta_only_model()
    model.compile(
        optimizer=keras.optimizers.AdamW(1e-4),
        loss=keras.losses.CategoricalCrossentropy(),
        metrics=[keras.metrics.F1Score(average="macro",name="f1")]
    )
    model.fit(ds_train, validation_data=ds_valid, epochs=EPOCHS,callbacks = callbacks)
    K.clear_session()

import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

def plot_training_history(metrics_logger):
    history = metrics_logger.history
    
    epochs = range(1, len(history["loss"]) + 1)
    
    # Loss vs Epoch
    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Loss vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("Loss")
    plt.legend()
    
    # F1 Score vs Epoch
    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["f1"], label="Training F1 Score")
    plt.plot(epochs, history["val_f1"], label="Validation F1 Score")
    plt.title("F1 Score vs Epochs")
    plt.xlabel("Epochs")
    plt.ylabel("F1 Score")
    plt.legend()
    plt.savefig(f"./results/test_meta_only_{VERSION}.png")

plot_training_history(metrics_logger)

from sklearn.preprocessing import label_binarize

def plot_roc_curve(y_true, y_pred, labels, num_classes):
    y_true_bin = label_binarize(y_true, classes=labels) 
    plt.figure(figsize=(10, 10))

    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {labels[i]} (AUC = {roc_auc:.2f})")

    plt.plot([0, 1], [0, 1], "k--")  # 대각선 기준선
    plt.xlim([0.0, 1.0])
    plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate")
    plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve")
    plt.legend(loc="lower right")
    plt.savefig(f"./results/roc_meta_only_VERSION.png")

ds_test = tf.data.Dataset.from_tensor_slices(X_test).batch(BATCH_SIZE).prefetch(AUTOTUNE)

y_preds = []
for cpkt in tqdm(glob(f"./ckpts/keras/{VERSION}/fold_*.keras")):        
    best_model = keras.models.load_model(cpkt, compile=False)
    y_preds.append(best_model.predict(ds_test, verbose=0))
    
    K.clear_session()

# 예측
y_preds = np.sum(np.array(y_preds), axis=0)
plot_roc_curve(y_test,y_preds,labels,len(labels))
y_preds = ohe.inverse_transform(y_preds).reshape(-1)

# 원래 y_test도 원래 라벨 복원
#y_true = ohe.inverse_transform(y_test).reshape(-1)

# 결과 저장
df_submission = pd.DataFrame({
    'pred': y_preds,
    'true': y_test
})
df_submission.to_csv("./results/keras/{VERSION}.csv", index=False)

