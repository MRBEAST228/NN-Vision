
# ============================================================
# 1. IMPORTS И НАСТРОЙКИ
# ============================================================

import os
import numpy as np
import tensorflow as tf
import matplotlib.pyplot as plt

from tensorflow.keras import layers, models, regularizers
from tensorflow.keras.preprocessing.image import ImageDataGenerator
from tensorflow.keras.callbacks import (
    EarlyStopping,
    ReduceLROnPlateau,
    ModelCheckpoint
)
from sklearn.metrics import classification_report, confusion_matrix

DATASET_PATH = "dataset/"
IMG_SIZE = (128, 128)
BATCH_SIZE = 32
EPOCHS = 60
SEED = 42

tf.keras.utils.set_random_seed(SEED)

if not os.path.isdir(DATASET_PATH):
    raise FileNotFoundError(f"Dataset not found: {DATASET_PATH}")

# ============================================================
# 2. ПОДГОТОВКА ДАННЫХ
# ============================================================

# Аугментация применяется только к тренировочным изображениям.
# Валидационные изображения остаются без изменений.

train_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.20,
    rotation_range=12,
    width_shift_range=0.08,
    height_shift_range=0.08,
    zoom_range=0.12,
    horizontal_flip=True
)

val_datagen = ImageDataGenerator(
    rescale=1.0 / 255,
    validation_split=0.20
)

train_data = train_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="training",
    shuffle=True,
    seed=SEED
)

val_data = val_datagen.flow_from_directory(
    DATASET_PATH,
    target_size=IMG_SIZE,
    batch_size=BATCH_SIZE,
    class_mode="categorical",
    subset="validation",
    shuffle=False
)

num_classes = len(train_data.class_indices)
class_names = list(train_data.class_indices.keys())

if num_classes < 2:
    raise ValueError("Dataset must contain at least 2 classes.")

if train_data.samples == 0 or val_data.samples == 0:
    raise ValueError("Not enough images for training/validation.")

print("\nClass mapping:", train_data.class_indices)
print("Number of classes:", num_classes)
print("Training images:", train_data.samples)
print("Validation images:", val_data.samples)

# ============================================================
# 3. ПРОВЕРКА БАЛАНСА КЛАССОВ
# ============================================================

counts = np.bincount(
    train_data.classes,
    minlength=num_classes
)

print("\nImages per class:")
for name, index in train_data.class_indices.items():
    print(f"{name}: {counts[index]}")

# Вес каждого класса помогает при дисбалансе данных.
class_weights = {
    i: train_data.samples / (num_classes * counts[i])
    for i in range(num_classes)
    if counts[i] > 0
}


# ============================================================
# 4. ОБЛЕГЧЁННАЯ CNN
# ============================================================

inputs = layers.Input(shape=(*IMG_SIZE, 3))

# Блок 1
x = layers.Conv2D(16, 3, padding="same",
                  kernel_initializer="he_normal")(inputs)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D()(x)

# Блок 2
x = layers.Conv2D(32, 3, padding="same",
                  kernel_initializer="he_normal")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D()(x)

# Блок 3
x = layers.Conv2D(64, 3, padding="same",
                  kernel_initializer="he_normal")(x)
x = layers.BatchNormalization()(x)
x = layers.Activation("relu")(x)
x = layers.MaxPooling2D()(x)

# Классификатор
x = layers.GlobalAveragePooling2D()(x)
x = layers.Dense(64, activation="relu")(x)
x = layers.Dropout(0.30)(x)

outputs = layers.Dense(num_classes, activation="softmax")(x)

model = models.Model(inputs=inputs, outputs=outputs)

model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=0.0003),
    loss="categorical_crossentropy",
    metrics=["accuracy"]
)

model.summary()

# ============================================================
# 5. КОМПИЛЯЦИЯ
# ============================================================

model.summary()

# ============================================================
# 6. CALLBACKS
# ============================================================

callbacks = [
    ModelCheckpoint(
        "best_cnn.keras",
        monitor="val_loss",
        save_best_only=True,
        mode="min",
        verbose=1
    ),

    EarlyStopping(
        monitor="val_loss",
        patience=10,
        restore_best_weights=True,
        verbose=1
    ),

    ReduceLROnPlateau(
        monitor="val_loss",
        factor=0.5,
        patience=3,
        min_lr=1e-7,
        verbose=1
    )
]

# ============================================================
# 7. ОБУЧЕНИЕ
# ============================================================

history = model.fit(
    train_data,
    validation_data=val_data,
    epochs=EPOCHS,
    callbacks=callbacks,
    class_weight=class_weights
)

# ============================================================
# 8. ЗАГРУЗКА ЛУЧШЕЙ МОДЕЛИ
# ============================================================

model = tf.keras.models.load_model("best_cnn.keras")

# ============================================================
# 9. ОЦЕНКА ТОЧНОСТИ
# ============================================================

val_loss, val_acc = model.evaluate(
    val_data,
    verbose=1
)

print("\n===================================")
print(f"Validation loss: {val_loss:.4f}")
print(f"Validation accuracy: {val_acc * 100:.2f}%")
print("===================================")

# ============================================================
# 10. ОТЧЁТ ПО КЛАССАМ
# ============================================================

val_data.reset()

probabilities = model.predict(val_data, verbose=1)
predictions = np.argmax(probabilities, axis=1)
true_labels = val_data.classes

print("\nCLASSIFICATION REPORT:\n")

print(
    classification_report(
        true_labels,
        predictions,
        labels=list(range(num_classes)),
        target_names=class_names,
        zero_division=0
    )
)

print("\nCONFUSION MATRIX:\n")
print(confusion_matrix(true_labels, predictions))

# ============================================================
# 11. ГРАФИКИ ОБУЧЕНИЯ
# ============================================================

plt.figure(figsize=(12, 5))

plt.subplot(1, 2, 1)
plt.plot(history.history["accuracy"], label="Train accuracy")
plt.plot(history.history["val_accuracy"], label="Validation accuracy")
plt.title("CNN Accuracy")
plt.xlabel("Epoch")
plt.ylabel("Accuracy")
plt.legend()
plt.grid(True)

plt.subplot(1, 2, 2)
plt.plot(history.history["loss"], label="Train loss")
plt.plot(history.history["val_loss"], label="Validation loss")
plt.title("CNN Loss")
plt.xlabel("Epoch")
plt.ylabel("Loss")
plt.legend()
plt.grid(True)

plt.tight_layout()
plt.savefig("cnn_training_history.png", dpi=150)
plt.show()

# ============================================================
# 12. СОХРАНЕНИЕ
# ============================================================

model.save("image_classifier_cnn.keras")

print("\nCNN saved successfully!")
print("Model: image_classifier_cnn.keras")
print("Best checkpoint: best_cnn.keras")
