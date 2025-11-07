# -*- coding: utf-8 -*-

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
import math
from tensorflow.keras import layers

from tqdm import tqdm
from keras_cv_attention_models import swin_transformer_v2
from keras_cv_attention_models import convnext
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.model_selection import StratifiedKFold

from sklearn.preprocessing import OneHotEncoder
from IPython.display import clear_output

import datetime as dt
import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc
from sklearn.preprocessing import label_binarize

# =========================
# Util
# =========================
def generate_output_filename():
    now = dt.datetime.now()
    return now.strftime("%Y%m%d_%H%M")

SEED = 42
IMG_SIZE = 98
BATCH_SIZE = 8
EPOCHS = 200 
NUM_CLASSES = 3
NUM_META_INPUTS = 4
STAMP = generate_output_filename()
VERSION = f"ENSEMBLE_{STAMP}"
AUTOTUNE = tf.data.AUTOTUNE

keras.utils.set_random_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CUDNN_DETERMINISTIC'] = "1"
os.environ['TF_USE_CUDNN'] = "true"

# =========================
# Data
# =========================
df = pd.read_csv("./rm_invalid.csv")
df = df.sample(frac=1, random_state=10).reset_index(drop=True)
df['ImagePath'] = df['ImagePath'].str.replace('/eos/home-y/yeo/ImageDataMergedEle/', './data/', regex=True)
df['dPhi'] = df['dPhi'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
df['dEta'] = df['dEta'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
df['dEtadPhi'] = df.apply(lambda row: np.concatenate((row['dEta'], row['dPhi'])), axis=1)

df_mergedHard = df[df['Label'] == 'mergedHard'].reset_index(drop=True).iloc[:30000]
df_notMerged = df[df['Label'] == 'notMerged'].reset_index(drop=True).iloc[:30000]
df_notElectron = df[df['Label'] == 'notElectron'].reset_index(drop=True).iloc[:30000]
df_train = pd.concat([df_mergedHard, df_notMerged, df_notElectron], ignore_index=True)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)

X = df_train["ImagePath"].to_numpy()
y = df_train["Label"].to_numpy()
meta = df_train['dEtadPhi'].to_numpy()
labels = np.unique(y)

ohe = OneHotEncoder(sparse_output=False)
ohe.fit(y.reshape(-1, 1))

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

def apply_augment(image, label=None):
    # placeholder (현재 augment 없음)
    if label is None:
        return image
    return (image, label)

X_train, X_test, meta_train, meta_test, y_train, y_test = train_test_split(X, meta, y, test_size=0.2, random_state=SEED, stratify=y)
skf = StratifiedKFold(n_splits=5, random_state=SEED, shuffle=True)
def preprocess(image, label):
    image, label = read_image(image, label)
    image, label = resize(image, label)
    image, label = apply_augment(image, label)
    return image, label
#X_train, X_valid, meta_train, meta_valid, y_train, y_valid = train_test_split(X_train, meta_train, y_train, test_size=0.25, random_state=SEED, stratify=y_train)

#print("A print site:", id(ds_test), ds_test.element_spec)
# predict 직전


#print(ds_test.element_spec)
#print("#############3333")
# =========================
# Models
# =========================
def build_image_model(name="image_backbone"):
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
    model = keras.Model(inputs, outputs, name=name)
    return model

def build_meta_model(name="meta_backbone"):
    meta_inputs = keras.Input((NUM_META_INPUTS,), name="meta_input")
    x = keras.layers.Dense(32, activation="relu")(meta_inputs)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(x)
    model = keras.Model(inputs=meta_inputs, outputs=outputs, name=name)
    return model

class DisplayCallback(keras.callbacks.Callback):
    def __init__(self, every=5):
        super().__init__()
        self.every = int(every)
    def on_epoch_end(self, epoch, logs=None):
        if epoch % self.every == 0:
            clear_output(wait=True)
    def get_config(self):
        return {"every": self.every}
    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)

def build_model(image_model: keras.Model, meta_model: keras.Model):
    keras.mixed_precision.set_global_policy("mixed_float16")
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    meta_inputs = keras.Input((NUM_META_INPUTS,))

    image_model.trainable = False
    meta_model.trainable = False

    out_i = image_model(inputs)
    out_m = meta_model(meta_inputs)

    # Trainable alpha via logistic bias init
    alpha0 = 0.5
    alpha_logit0 = math.log(alpha0 / (1.0 - alpha0))
    ones = tf.ones_like(out_i[..., :1])  # (B,1)
    alpha_layer = layers.Dense(
        1,
        activation="sigmoid",
        kernel_initializer="zeros",
        bias_initializer=keras.initializers.Constant(alpha_logit0),
        name="alpha_scalar",
    )
    alpha = alpha_layer(ones)  # (B,1)

    alpha = tf.cast(alpha, tf.float32)
    out_i = tf.cast(out_i, tf.float32)
    out_m = tf.cast(out_m, tf.float32)
    p_ens = alpha * out_m + (1.0 - alpha) * out_i
    out = p_ens

    ens_model = keras.Model(inputs=[inputs, meta_inputs], outputs=out, name="ensemble_model")
    return ens_model

class MetricsLogger(keras.callbacks.Callback):
    def __init__(self, keep_preds=False, max_store=0):
        super().__init__()
        self.keep_preds = bool(keep_preds)
        self.max_store = int(max_store)
        self.history = {"loss": [], "val_loss": [], "f1": [], "val_f1": [], "alpha": []}
        self.y_true, self.y_pred = [], []
    def on_epoch_end(self, epoch, logs=None):
        logs = logs or {}
        self.history["loss"].append(logs.get("loss"))
        self.history["val_loss"].append(logs.get("val_loss"))
        self.history["f1"].append(logs.get("f1"))
        self.history["val_f1"].append(logs.get("val_f1"))
        self.history["alpha"].append(logs.get("alpha_scalar"))
    def get_config(self):
        return {"keep_preds": self.keep_preds, "max_store": self.max_store}
    @classmethod
    def from_config(cls, cfg):
        return cls(**cfg)

os.makedirs(f"./ckpts/keras/{VERSION}", exist_ok=True)
metrics_logger = MetricsLogger()

# =========================
# Callbacks (가중치만 저장)
# =========================
for fold, (train_idx,valid_idx) in enumerate(skf.split(X_tmp,meta_tmp,y_tmp)):
    y_train = ohe.transform(y_train.reshape(-1, 1))
    y_valid = ohe.transform(y_valid.reshape(-1, 1))
    meta_train = np.stack(meta_train).astype(np.float32)
    meta_valid = np.stack(meta_valid).astype(np.float32)
    meta_test = np.stack(meta_test).astype(np.float32)
    
    # Datasets
    ds_train_image_label = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train), seed=SEED)
    ds_valid_image_label = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))


    ds_train_image_label = ds_train_image_label.map(preprocess, num_parallel_calls=AUTOTUNE).cache()
    ds_valid_image_label = ds_valid_image_label.map(
        lambda img, lbl: resize(*read_image(img, lbl)),
        num_parallel_calls=AUTOTUNE
    ).cache()

    ds_meta_train = tf.data.Dataset.from_tensor_slices(meta_train)
    ds_meta_valid = tf.data.Dataset.from_tensor_slices(meta_valid)

    ds_train_image = ds_train_image_label.map(lambda image, label: image)
    ds_train_label = ds_train_image_label.map(lambda image, label: label)
    ds_valid_image = ds_valid_image_label.map(lambda image, label: image)
    ds_valid_label = ds_valid_image_label.map(lambda image, label: label)

    ds_train = tf.data.Dataset.zip(((ds_train_image, ds_meta_train), ds_train_label))
    ds_valid = tf.data.Dataset.zip(((ds_valid_image, ds_meta_valid), ds_valid_label))
    ds_train = ds_train.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    ds_valid = ds_valid.batch(BATCH_SIZE).prefetch(AUTOTUNE)
    CKPT_PATH = f"./ckpts/keras/{VERSION}/fold_{fold}"  # 확장자 없이
    #CKPT_PATH = f"./ckpts/keras/ENSEMBLE_20251024_1753/best_weights"  # 확장자 없이
    
    callbacks = [
        DisplayCallback(),
        keras.callbacks.TensorBoard(log_dir=f"./logs/keras/{VERSION}/fold_{fold}"),
        keras.callbacks.EarlyStopping(monitor="val_f1", mode="max", verbose=0, patience=10),
        keras.callbacks.ModelCheckpoint(
            filepath=CKPT_PATH,
            monitor="val_f1", mode="max",
            save_best_only=True,
            save_weights_only=True,  # 핵심: 직렬화 우회
        ),
        keras.callbacks.ReduceLROnPlateau(monitor="val_f1", mode="max", factor=0.8, patience=5),
        metrics_logger
    ]
    
    # =========================
    # Build & Train
    # =========================
    #optimizer = keras.optimizers.AdamW(1e-5)
    #loss = keras.losses.CategoricalFocalCrossentropy(from_logits=False)
    #f1 = keras.metrics.F1Score(average="macro", name="f1")
    def make_optimizer():
        return keras.optimizers.AdamW(1e-5)
    
    def make_metrics():
        return [keras.metrics.F1Score(average="macro", name="f1")]
    
    loss = keras.losses.CategoricalFocalCrossentropy(from_logits=False)
    
    # 백본 로드 시 compile=False (직렬화 경로 깔끔)
    img_loaded  = keras.models.load_model(f'./ckpts/keras/wo_meta_20251022_0643/fold_{fold}.keras', compile=False)
    meta_loaded = keras.models.load_model(f'./ckpts/keras/only_meta_20251022_0212/fold_{fold}.keras', compile=False)
    
    image_backbone = keras.Model(img_loaded.inputs,  img_loaded.outputs,  name="image_backbone")
    meta_backbone  = keras.Model(meta_loaded.inputs, meta_loaded.outputs, name="meta_backbone")
    
    model = build_model(image_backbone, meta_backbone)
    model.compile(optimizer=make_optimizer(), loss=loss, metrics=make_metrics())
    
    print(f"Train/Valid sizes: {len(X_train)} / {len(X_valid)}  |  Test: {len(X_test)}")
    model.fit(ds_train, validation_data=ds_valid, epochs=EPOCHS, callbacks=callbacks)
    K.clear_session()

# =========================
# Reload best & Evaluate/Predict
# =========================
# 동일 구조로 모델 재생성 후 최적 가중치 로드
y_preds = []
for cpkt in tqdm(glob(f"./ckpts/keras/{VERSION}/fold_*")):        
    best_model = build_model(image_backbone, meta_backbone)
    best_model.compile(optimizer=make_optimizer(), loss=loss, metrics=make_metrics())
    best_model.load_weights(CKPT_PATH)

    best_model.evaluate(ds_valid, verbose=1)
    y_preds.append(best_model.predict(ds_test, verbose=0))
    
    K.clear_session()

y_preds = np.sum(np.array(y_preds), axis=0)

# 평가 (원하면)

# Test dataset (predict 용: 라벨 없음)
ds_test_image = tf.data.Dataset.from_tensor_slices(X_test)
ds_test_image = ds_test_image.map(lambda image: read_image(image), num_parallel_calls=AUTOTUNE)
ds_test_image = ds_test_image.map(lambda image: resize(image), num_parallel_calls=AUTOTUNE)
ds_meta_test = tf.data.Dataset.from_tensor_slices(meta_test)
ds_test = tf.data.Dataset.zip((ds_test_image, ds_meta_test)).map(lambda img, meta: ((img, meta),), num_parallel_calls=AUTOTUNE)
ds_test = ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE)
# =========================
# Plots & Submission
# =========================
def plot_training_history(metrics_logger, version):
    history = metrics_logger.history
    epochs = range(1, len(history["loss"]) + 1)

    plt.figure(figsize=(12, 6))
    plt.subplot(1, 2, 1)
    plt.plot(epochs, history["loss"], label="Training Loss")
    plt.plot(epochs, history["val_loss"], label="Validation Loss")
    plt.title("Loss vs Epochs"); plt.xlabel("Epochs"); plt.ylabel("Loss"); plt.legend()

    plt.subplot(1, 2, 2)
    plt.plot(epochs, history["f1"], label="Training F1")
    plt.plot(epochs, history["val_f1"], label="Validation F1")
    plt.title("F1 vs Epochs"); plt.xlabel("Epochs"); plt.ylabel("F1"); plt.legend()

    os.makedirs("./results", exist_ok=True)
    plt.savefig(f"./results/test_{version}.png", bbox_inches="tight")
    plt.close()

plot_training_history(metrics_logger, VERSION)

def plot_training_history_alpha(metrics_logger,version):
    history = metrics_logger.history
    epochs = range(1, len(history["loss"]) + 1)
    plt.figure(figsize=(6, 6))
    plt.plot(epochs, history["alpha"],label='training value')
    plt.xlabel('Epochs')
    plt.ylabel('alpha')
    os.makedirs("./results", exist_ok=True)
    plt.savefig(f"./results/test_{version}.png", bbox_inches="tight")
    
plot_training_history_alpha(metrics_logger,VERSION)


def plot_roc_curve(y_true_raw, y_pred_probs, label_names, num_classes, version):
    y_true_bin = label_binarize(y_true_raw, classes=label_names)
    plt.figure(figsize=(10, 10))
    for i in range(num_classes):
        fpr, tpr, _ = roc_curve(y_true_bin[:, i], y_pred_probs[:, i])
        roc_auc = auc(fpr, tpr)
        plt.plot(fpr, tpr, label=f"Class {label_names[i]} (AUC = {roc_auc:.2f})")
    plt.plot([0, 1], [0, 1], "k--")
    plt.xlim([0.0, 1.0]); plt.ylim([0.0, 1.05])
    plt.xlabel("False Positive Rate"); plt.ylabel("True Positive Rate")
    plt.title("Multi-Class ROC Curve"); plt.legend(loc="lower right")
    os.makedirs("./results", exist_ok=True)
    plt.savefig(f"./results/roc_{version}.png", bbox_inches="tight")
    plt.close()

# 예측은 best_model로
print("Before predict:", id(ds_test), ds_test.element_spec)
plot_roc_curve(y_test, y_preds, labels, len(labels), VERSION)
y_preds = ohe.inverse_transform(y_preds).reshape(-1)

# 최종 CSV
os.makedirs(f"./results/keras", exist_ok=True)
pd.DataFrame({"pred": y_preds, "true": y_test}).to_csv(f"./results/keras/{VERSION}.csv", index=False)


