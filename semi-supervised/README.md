"""
ResNet50 on CIFAR-10 (Transfer Learning)
----------------------------------------
- Loads CIFAR-10 dataset
- Uses pretrained ResNet50 (ImageNet)
- Trains a custom classification head
"""

import tensorflow as tf
from tensorflow.keras.applications import ResNet50
from tensorflow.keras.datasets import cifar10
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, GlobalAveragePooling2D
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.utils import to_categorical


# =========================
# Data Loading & Preprocessing
# =========================
def load_data():
    (x_train, y_train), (x_test, y_test) = cifar10.load_data()

    # Normalize images
    x_train = x_train.astype("float32") / 255.0
    x_test = x_test.astype("float32") / 255.0

    # One-hot encode labels
    y_train = to_categorical(y_train, 10)
    y_test = to_categorical(y_test, 10)

    return x_train, y_train, x_test, y_test


# =========================
# Model Building
# =========================
def build_model(input_shape=(32, 32, 3), num_classes=10):
    base_model = ResNet50(
        weights="imagenet",
        include_top=False,
        input_shape=input_shape
    )

    # Freeze pretrained layers
    base_model.trainable = False

    model = Sequential([
        base_model,
        GlobalAveragePooling2D(),
        Dense(1024, activation="relu"),
        Dense(num_classes, activation="softmax")
    ])

    return model


# =========================
# Training
# =========================
def train_model(model, x_train, y_train, x_test, y_test):
    model.compile(
        optimizer=Adam(learning_rate=1e-4),
        loss="categorical_crossentropy",
        metrics=["accuracy"]
    )

    history = model.fit(
        x_train, y_train,
        batch_size=64,
        epochs=5,
        validation_data=(x_test, y_test)
    )

    return history


# =========================
# Evaluation
# =========================
def evaluate_model(model, x_test, y_test):
    test_loss, test_acc = model.evaluate(x_test, y_test)
    print(f"\nTest Accuracy: {test_acc:.4f}")
    return test_loss, test_acc


# =========================
# Main Execution
# =========================
def main():
    x_train, y_train, x_test, y_test = load_data()

    model = build_model()
    model.summary()

    train_model(model, x_train, y_train, x_test, y_test)
    evaluate_model(model, x_test, y_test)


if __name__ == "__main__":
    main()
