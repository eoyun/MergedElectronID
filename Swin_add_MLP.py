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

from tqdm import tqdm
from keras_cv_attention_models import swin_transformer_v2
from keras_cv_attention_models import convnext
from glob import glob
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import OneHotEncoder
from IPython.display import clear_output

import datetime

def generate_output_filename():
    now = datetime.datetime.now()
    timestamp = now.strftime("%Y%m%d_%H%M")
    filename = f"{timestamp}"
    return filename

SEED = 42
IMG_SIZE = 98
BATCH_SIZE = 8
EPOCHS = 1000
NUM_CLASSES = 3
NUM_META_INPUTS = 4
datetime = generate_output_filename()
VERSION = f"SWIN_meta_{datetime}"
AUTOTUNE = tf.data.AUTOTUNE

keras.utils.set_random_seed(SEED)
os.environ['PYTHONHASHSEED'] = str(SEED)
os.environ["CUDA_VISIBLE_DEVICES"] = "1"
os.environ['TF_CUDNN_DETERMINISTIC'] = "1"
os.environ['TF_USE_CUDNN'] = "true"

df = pd.read_csv("./data/250408_v1.csv")
df = df.sample(frac=1, random_state=10).reset_index(drop=True)
df['ImagePath'] = df['ImagePath'].str.replace('/eos/home-y/yeo/ImageDataMergedEle/', './data/', regex=True)  
df['dPhi'] = df['dPhi'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
df['dEta'] = df['dEta'].apply(lambda x: np.fromstring(x.strip("[]"), sep=' '))
df['dEtadPhi'] = df.apply(lambda row: np.concatenate((row['dEta'], row['dPhi'])), axis=1)

df_mergedHard = df[df['Label'] == 'mergedHard'].reset_index(drop=True)
df_mergedHard = df_mergedHard.iloc[:30000]
df_notMerged = df[df['Label'] == 'notMerged'].reset_index(drop=True)
df_notMerged = df_notMerged.iloc[:30000]
df_notElectron = df[df['Label'] == 'notElectron'].reset_index(drop=True)
df_notElectron = df_notElectron.iloc[:30000]
df_train = pd.concat([df_mergedHard, df_notMerged, df_notElectron], ignore_index=True)
df_train = df_train.sample(frac=1, random_state=42).reset_index(drop=True)
print(df_train)

X = df_train["ImagePath"].to_numpy()
y = df_train["Label"].to_numpy()

meta = df_train['dEtadPhi'].to_numpy()
print(X)
print(y)
print(meta)
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

def build_model(normalizer):
    keras.mixed_precision.set_global_policy("mixed_float16")
    inputs = keras.Input((IMG_SIZE, IMG_SIZE, 3))
    meta_inputs = keras.Input((NUM_META_INPUTS,))

    backbone = swin_transformer_v2.SwinTransformerV2Base_window16(
    #backbone = convnext.ConvNeXtBase(
        input_shape=(IMG_SIZE, IMG_SIZE, 3),
        num_classes=0,
        pretrained="imageNet21k-ft1k"
    )
    x = keras.layers.Rescaling(scale=1./127.5, offset=-1.)(inputs)
    x = backbone(x)
    x = keras.layers.GlobalAveragePooling2D()(x)
    x = keras.layers.Dense(512, activation="relu")(x)
    x = keras.layers.Dense(256, activation="relu")(x)
    x = keras.layers.Dense(128, activation="relu")(x)
    x = keras.layers.Dense(64, activation="relu")(x)
    x = keras.layers.Dense(32, activation="relu")(x)
    x = keras.layers.Dense(16, activation="relu")(x)
    x = keras.layers.Dense(12, activation="relu")(x)

    y = normalizer(meta_inputs)
    #y = keras.layers.Dense(4, activation="relu")(y)  # MLP 1층
    #y = keras.layers.Dense(4, activation="relu")(y)  # MLP 2층
    #y = keras.layers.Dropout(0.2)(y)  # MLP 3�Dropout

    combined = keras.layers.Concatenate()([x, y])
    #combined = keras.layers.Dense(128, activation="relu")(combined)
    #combined = keras.layers.Dense(64, activation="relu")(combined)
    #combined = keras.layers.Dense(64, activation="relu")(combined)

    outputs = keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(combined)

    model = keras.Model(inputs=[inputs, meta_inputs], outputs=outputs)
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
def build_debug_model():
    image_input = keras.Input(shape=(IMG_SIZE, IMG_SIZE, 3), name="image_input")

    meta_input = keras.Input(shape=(NUM_META_INPUTS,), name="meta_input")

    x = keras.layers.Rescaling(1./255)(image_input)
    x = keras.layers.Conv2D(16, 3, activation='relu')(x)
    x = keras.layers.GlobalAveragePooling2D()(x)

    combined = keras.layers.Concatenate()([x, meta_input])

    output = keras.layers.Dense(NUM_CLASSES, activation="softmax", dtype="float32")(combined)

    model = keras.Model(inputs=[image_input, meta_input], outputs=output)
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
X_train, X_test, meta_train, meta_test, y_train, y_test = train_test_split(X, meta, y, test_size =0.2)
X_train, X_valid, meta_train, meta_valid, y_train, y_valid = train_test_split(X_train, meta_train, y_train, test_size =0.5)
print(f"{len(X_test)} : test")
print(f"{len(X_train)} : train")
print(f"{len(X_valid)} : valid")
y_train = ohe.transform(y_train.reshape(-1, 1))
y_valid = ohe.transform(y_valid.reshape(-1, 1))
meta_train = np.stack(meta_train).astype(np.float32)
meta_valid = np.stack(meta_valid).astype(np.float32)
meta_test = np.stack(meta_test).astype(np.float32)
print(type(meta_test))           
print(meta_test.shape)           
print(meta_test[0])              
print(type(meta_test[0]))
normalizer = tf.keras.layers.Normalization()
ds_test_image = tf.data.Dataset.from_tensor_slices(X_test)
ds_test_image = ds_test_image.map(lambda image: read_image(image), num_parallel_calls=AUTOTUNE)
ds_test_image = ds_test_image.map(lambda image: resize(image), num_parallel_calls=AUTOTUNE)

ds_meta_test = tf.data.Dataset.from_tensor_slices(meta_test)

ds_test = tf.data.Dataset.zip((ds_test_image, ds_meta_test))
ds_test = ds_test.map(lambda img, meta: ((img, meta)), num_parallel_calls=AUTOTUNE)

ds_test = ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE).cache()
print(ds_test)
#ds_train = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train))
#ds_train = ds_train.map(lambda image, label: read_image(image, label), num_parallel_calls=AUTOTUNE).cache()
#ds_train = ds_train.map(lambda image, label: resize(image, label), num_parallel_calls=AUTOTUNE)
#ds_train = ds_train.map(lambda image, label: apply_augment(image, label), num_parallel_calls=AUTOTUNE)
#ds_meta_train = tf.data.Dataset.from_tensor_slices(meta_train)
#ds_train = ds_train.batch(BATCH_SIZE)
#d
#ds_valid = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))
#ds_valid = ds_valid.map(lambda image, label: read_image(image, label), num_parallel_calls=AUTOTUNE)
#ds_valid = ds_valid.map(lambda image, label: resize(image, label), num_parallel_calls=AUTOTUNE)
#ds_meta_valid = tf.data.Dataset.from_tensor_slices(meta_valid)
#ds_valid = ds_valid.batch(BATCH_SIZE).prefetch(AUTOTUNE).cache()

ds_train_image_label = tf.data.Dataset.from_tensor_slices((X_train, y_train)).shuffle(len(y_train))
ds_valid_image_label = tf.data.Dataset.from_tensor_slices((X_valid, y_valid))

def preprocess(image, label):
    image, label = read_image(image, label)
    image, label = resize(image, label)
    image, label = apply_augment(image, label)
    return image, label

ds_train_image_label = ds_train_image_label.map(preprocess, num_parallel_calls=AUTOTUNE).cache()
ds_valid_image_label = ds_valid_image_label.map(lambda img, lbl: resize(*read_image(img, lbl)), num_parallel_calls=AUTOTUNE).cache()

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

normalizer.adapt(ds_meta_train)
# Cosine Annealing + Warm Restarts
initial_learning_rate = 1e-4
first_decay_steps = 25  # 첫번째 주기
t_mul = 2.0  # 주기 확장 배율 (다음 주기는 200, 400, ... 으로 늘어남)
m_mul = 1.0  # 최대 LR 감소 비율 (1.0이면 그대로 유지)
alpha = 0.0  # 최소 learning rate 비율

lr_schedule = keras.optimizers.schedules.CosineDecayRestarts(
    initial_learning_rate,
    first_decay_steps,
    t_mul=t_mul,
    m_mul=m_mul,
    alpha=alpha
)


callbacks = [
    DisplayCallback(),
    keras.callbacks.TensorBoard(log_dir=f"./logs/keras/{VERSION}/test"),
    keras.callbacks.EarlyStopping(monitor="val_f1", mode="max", verbose=0, patience=5),
    keras.callbacks.ModelCheckpoint(f"./ckpts/keras/{VERSION}/test.keras", monitor="val_f1", mode="max", save_best_only=True),
    keras.callbacks.ReduceLROnPlateau(monitor="val_f1", mode="min", factor=0.8, patience=3),
    #keras.callbacks.ReduceLROnPlateau(monitor="val_f1", mode="min", patience=3),
    metrics_logger
]

optimizer = keras.optimizers.AdamW(1e-5)
#optimizer = keras.optimizers.SGD(1e-5)
#optimizer = keras.optimizers.SGD(learning_rate = lr_schedule, momentum = 0.9)
loss = keras.losses.CategoricalFocalCrossentropy(from_logits=False)
f1 = keras.metrics.F1Score(average="macro", name="f1")

model = build_model(normalizer)
#model = build_debug_model()
model.compile(optimizer=optimizer, loss=loss, metrics=[f1])
model.fit(ds_train, validation_data=ds_valid , epochs=EPOCHS, callbacks=callbacks)

K.clear_session()


import matplotlib.pyplot as plt
from sklearn.metrics import roc_curve, auc, RocCurveDisplay

def plot_training_history(metrics_logger,VERSION):
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
    plt.savefig(f"test_{VERSION}.png")

plot_training_history(metrics_logger,VERSION)

from sklearn.preprocessing import label_binarize

def plot_roc_curve(y_true, y_pred, labels, num_classes, VERSION):
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
    plt.savefig(f"roc_{VERSION}.png")

#ds_test = tf.data.Dataset.from_tensor_slices(X_test)
#ds_test = ds_test.map(lambda image: read_image(image), num_parallel_calls=AUTOTUNE)
#ds_test = ds_test.map(lambda image: resize(image), num_parallel_calls=AUTOTUNE)
#ds_meta_test = tf.data.Dataset.from_tensor_slices(meta_test)
#ds_test = ds_test.batch(BATCH_SIZE*2).prefetch(AUTOTUNE).cache()
ds_test_image = tf.data.Dataset.from_tensor_slices(X_test)
ds_test_image = ds_test_image.map(lambda image: read_image(image), num_parallel_calls=AUTOTUNE)
ds_test_image = ds_test_image.map(lambda image: resize(image), num_parallel_calls=AUTOTUNE)

ds_meta_test = tf.data.Dataset.from_tensor_slices(meta_test)

ds_test = tf.data.Dataset.zip((ds_test_image, ds_meta_test))
ds_test = ds_test.map(lambda img, meta: ((img, meta),), num_parallel_calls=AUTOTUNE)

ds_test = ds_test.batch(BATCH_SIZE).prefetch(AUTOTUNE).cache()

y_preds = []
for cpkt in tqdm(glob(f"./ckpts/keras/{VERSION}/test.keras")):        
    best_model = keras.models.load_model(cpkt, compile=False)
    y_preds.append(best_model.predict(ds_test, verbose=0))
    
    K.clear_session()
print(y_preds)
y_preds = np.sum(np.array(y_preds), axis=0)

plot_roc_curve(y_test,y_preds,labels,len(labels),VERSION)

y_preds = ohe.inverse_transform(y_preds).reshape(-1)

df_submission = pd.DataFrame({
    'pred' : y_preds,
    'true' : y_test
    })
df_submission.to_csv(f"./results/keras/{VERSION}.csv", index=False)
