
import os
import warnings
warnings.filterwarnings("ignore")

from silence_tensorflow import silence_tensorflow
silence_tensorflow()

import numpy as np
import pandas as pd
import keras
import keras_cv
import tensorflow as tf
import keras.backend as K

from tqdm import tqdm
from keras_cv_attention_models import swin_transformer_v2
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from IPython.display import clear_output

SEED = 42
IMG_SIZE = 98
BATCH_SIZE = 16
EPOCHS = 100
NUM_CLASSES = 3
VERSION = "batch_test"
AUTOTUNE = tf.data.AUTOTUNE

keras.utils.set_random_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ['TF_CUDNN_DETERMINISTIC'] = "1"
os.environ['TF_USE_CUDNN'] = "true"

df = pd.read_csv("./data/250303_v1.csv")

df_mergedHard = df[df['Label'] == 'mergedHard'].reset_index(drop=True)
df_mergedHard = df_mergedHard.iloc[:20001]
df_notMerged = df[df['Label'] == 'notMerged'].reset_index(drop=True)
df_notMerged = df_notMerged.iloc[:20001]
df_notElectron = df[df['Label'] == 'notElectron'].reset_index(drop=True)
df_notElectron = df_notElectron.iloc[:20001]
df_train = pd.concat([df_mergedHard, df_notMerged, df_notElectron], ignore_index=True)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

X = df_train["ImagePath"].to_numpy()
y = df_train["Label"].to_numpy()

labels = np.unique(y)

def read_image(image_path, label=None):
    image = tf.io.read_file(image_path)
    image = tf.image.decode_jpeg(image, channels=3)
    image = tf.cast(image, tf.float32)

    if label is None:
        return image

    label = tf.cast(label, tf.float32)
    return (image, label)

def resize(image, label=None, size=IMG_SIZE):
    image = tf.image.resize(image, (size, size), "bicubic")

    if label is None:
        return image
    return (image, label)

zoom_out = keras_cv.layers.RandomZoom((0.1, 0.4))
zoom_in = keras_cv.layers.RandomZoom((-0.4, -0.1))

aug_layers = [
    keras_cv.layers.RandomApply(keras_cv.layers.RandomChoice([zoom_out, zoom_in])),
    keras_cv.layers.RandomApply(layer=keras_cv.layers.RandomRotation(factor=(-0.2, 0.2))),
    keras_cv.layers.RandomApply(layer=keras_cv.layers.RandomBrightness(factor=0.2)),
    keras_cv.layers.RandomApply(layer=keras_cv.layers.RandomContrast(value_range=(0, 255), factor=0.2)),
    keras_cv.layers.RandomApply(layer=keras_cv.layers.RandomShear(0.2, 0.2))
]

def apply_augment(image, label=None):
    for layer in aug_layers:
        image = layer(image)

    if label is None:
        return image
    return (image, label)



ohe = OneHotEncoder(sparse_output=False)
ohe.fit(y.reshape(-1, 1))

def build_model():
    keras.mixed_precision.set_global_policy("mixed_float16")
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))

    backbone = swin_transformer_v2.SwinTransformerV2Base_window16(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        num_classes=0,
        pretrained="imagenet22k"
    )
    x = keras.layers.Rescaling(scale=1./127.5, offset=-1.)(inputs)
    x = backbone(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)

    model = keras.Model(inputs, outputs)
    return model

class DisplayCallback(keras.callbacks.Callback):
    def on_epoch_end(self, epoch, logs=None):
        if epoch % 5 == 0:
            clear_output(wait=True)

def get_debug_swin_v2():
    model = swin_transformer_v2.SwinTransformerV2(
        input_shape=(98, 98, 3),  # 작은 입력 크기
        num_classes=3,  # 작은 출력 클래스 수
        embed_dim=32,  # 작은 임베딩 차원
        num_heads=(2, 4, 8),
        window_size=4,
    )
    return model

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


metrics_logger = MetricsLogger()
#skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
#for fold, (train_idx, valid_idx) in enumerate(skf.split(X, y)):
#X_train, y_train = X[train_idx], y[train_idx]
#X_valid, y_valid = X[valid_idx], y[valid_idx]
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size =0.2)
X_train, X_valid, y_train, y_valid = train_test_split(X_train, y_train, test_size =0.5)
print(f"{len(X_test)} : test")
print(f"{len(X_train)} : train")
print(f"{len(X_valid)} : valid")
y_train = ohe.transform(y_train.reshape(-1, 1))
y_valid = ohe.transform(y_valid.reshape(-1, 1))

ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train))
ds_train = ds_train.map(lambda image, label: read_image(image, label), num_parallel_calls=AUTOTUNE).cache()
ds_train = ds_train.map(lambda image, label: resize(image, label), num_parallel_calls=AUTOTUNE)
ds_train = ds_train.map(lambda image, label: apply_augment(image, label), num_parallel_calls=AUTOTUNE)
ds_train = ds_train.batch(BATCH_SIZE)

ds_valid = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
ds_valid = ds_valid.map(lambda image, label: read_image(image, label), num_parallel_calls=AUTOTUNE)
ds_valid = ds_valid.map(lambda image, label: resize(image, label), num_parallel_calls=AUTOTUNE)
ds_valid = ds_valid.batch(BATCH_SIZE).prefetch(AUTOTUNE).cache()


callbacks = [
    DisplayCallback(),
    keras.callbacks.TensorBoard(log_dir=f"./logs/keras/{VERSION}/test"),
    keras.callbacks.EarlyStopping(monitor="val_f1", mode="max", verbose=0, patience=5),
    keras.callbacks.ModelCheckpoint(f"./ckpts/keras/{VERSION}/test.keras", monitor="val_f1", mode="max", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_f1", mode="min", factor=0.8, patience=3),
    metrics_logger
]

optimizer = keras.optimizers.AdamW(1e-5)
loss = keras.losses.CategoricalFocalCrossentropy(from_logits=False)
f1 = keras.metrics.F1Score(average="macro", name="f1")

model = build_model()
#model = get_debug_swin_v2()
model.compile(optimizer=optimizer, loss=loss, metrics=[f1])
model.fit(ds_train, validation_data=ds_valid, epochs=EPOCHS, callbacks=callbacks)

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
    plt.savefig("test.png")

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
    plt.savefig("roc.png")

ds_test = tf.data.Dataset.from_tensor_slices(X_test)
ds_test = ds_test.map(lambda image: read_image(image), num_parallel_calls=AUTOTUNE)
ds_test = ds_test.map(lambda image: resize(image), num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE*2).prefetch(AUTOTUNE).cache()

y_preds = []
for cpkt in tqdm(glob(f"./ckpts/keras/{VERSION}/test.keras")):        
    best_model = keras.models.load_model(cpkt, compile=False)
    y_preds.append(best_model.predict(ds_test, verbose=0))
    
    K.clear_session()
print(y_preds)
y_preds = np.sum(np.array(y_preds), axis=0)

plot_roc_curve(y_test,y_preds,labels,len(labels))

y_preds = ohe.inverse_transform(y_preds).reshape(-1)

df_submission = pd.DataFrame({
    'pred' : y_preds,
    'true' : y_test
    })
df_submission.to_csv(f"./results/keras/{VERSION}.csv", index=False)
