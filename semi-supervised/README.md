{
  "cells": [
    {
      "cell_type": "markdown",
      "metadata": {
        "id": "view-in-github",
        "colab_type": "text"
      },
      "source": [
        "<a href=\"https://colab.research.google.com/github/kmyers111/ML-Models/blob/main/supervised/Resnet50CNN.ipynb\" target=\"_parent\"><img src=\"https://colab.research.google.com/assets/colab-badge.svg\" alt=\"Open In Colab\"/></a>"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "ktQV5i2i4IzP"
      },
      "outputs": [],
      "source": [
        "import tensorflow as tf\n",
        "from tensorflow.keras.applications import ResNet50\n",
        "from tensorflow.keras.datasets import cifar10\n",
        "from tensorflow.keras.preprocessing.image import ImageDataGenerator\n",
        "from tensorflow.keras.models import Sequential\n",
        "from tensorflow.keras.layers import Dense, GlobalAveragePooling2D\n",
        "from tensorflow.keras.optimizers import Adam\n",
        "from tensorflow.keras.utils import to_categorical"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "uL4P9jKQ4ceZ",
        "outputId": "e47bdd42-76ac-4ccd-bec5-ced5357fb52c"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Downloading data from https://www.cs.toronto.edu/~kriz/cifar-10-python.tar.gz\n",
            "\u001b[1m170498071/170498071\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m4s\u001b[0m 0us/step\n"
          ]
        }
      ],
      "source": [
        "# Load CIFAR-10 dataset\n",
        "(x_train, y_train), (x_test, y_test) = cifar10.load_data()\n",
        "\n",
        "# Preprocess the data\n",
        "x_train = x_train.astype('float32') / 255.0\n",
        "x_test = x_test.astype('float32') / 255.0\n",
        "\n",
        "# One-hot encode the labels\n",
        "y_train = to_categorical(y_train, 10)\n",
        "y_test = to_categorical(y_test, 10)"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "base_uri": "https://localhost:8080/"
        },
        "id": "ggiQ8BF04f9c",
        "outputId": "a077e223-3320-4aa0-ec6e-72ab6b8aa2cb"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Downloading data from https://storage.googleapis.com/tensorflow/keras-applications/resnet/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5\n",
            "\u001b[1m94765736/94765736\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m0s\u001b[0m 0us/step\n"
          ]
        }
      ],
      "source": [
        "base_model = ResNet50(weights='imagenet',\n",
        "                      include_top=False,\n",
        "                      input_shape=(32, 32, 3))\n",
        "\n",
        "# Freeze the base model\n",
        "base_model.trainable = False"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "9NfXCTj04kZu"
      },
      "outputs": [],
      "source": [
        "# Build the classification model\n",
        "model = Sequential([\n",
        "    base_model,\n",
        "    GlobalAveragePooling2D(),\n",
        "    Dense(1024, activation='relu'),\n",
        "    Dense(10, activation='softmax')\n",
        "])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "_vM1Lbap4o_X"
      },
      "outputs": [],
      "source": [
        "# Compile the model\n",
        "model.compile(optimizer=Adam(learning_rate=0.0001),\n",
        "              loss='categorical_crossentropy',\n",
        "              metrics=['accuracy'])"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "colab": {
          "background_save": true,
          "base_uri": "https://localhost:8080/"
        },
        "id": "3RU3KlYO4rPI",
        "outputId": "ddb7130e-98bf-44ae-fd8b-b88c06e2289c"
      },
      "outputs": [
        {
          "name": "stdout",
          "output_type": "stream",
          "text": [
            "Epoch 1/5\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m328s\u001b[0m 419ms/step - accuracy: 0.2469 - loss: 2.0721 - val_accuracy: 0.2728 - val_loss: 1.9822\n",
            "Epoch 2/5\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m287s\u001b[0m 367ms/step - accuracy: 0.2984 - loss: 1.9373 - val_accuracy: 0.3089 - val_loss: 1.8918\n",
            "Epoch 3/5\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m283s\u001b[0m 362ms/step - accuracy: 0.3272 - loss: 1.8742 - val_accuracy: 0.3490 - val_loss: 1.8331\n",
            "Epoch 4/5\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m358s\u001b[0m 408ms/step - accuracy: 0.3404 - loss: 1.8375 - val_accuracy: 0.3536 - val_loss: 1.8117\n",
            "Epoch 5/5\n",
            "\u001b[1m782/782\u001b[0m \u001b[32m━━━━━━━━━━━━━━━━━━━━\u001b[0m\u001b[37m\u001b[0m \u001b[1m279s\u001b[0m 357ms/step - accuracy: 0.3518 - loss: 1.8062 - val_accuracy: 0.3694 - val_loss: 1.7725\n"
          ]
        },
        {
          "data": {
            "text/plain": [
              "<keras.src.callbacks.history.History at 0x7fab0329ae10>"
            ]
          },
          "execution_count": 8,
          "metadata": {},
          "output_type": "execute_result"
        }
      ],
      "source": [
        "# Train the model\n",
        "model.fit(x_train, y_train,\n",
        "          batch_size=64,\n",
        "          epochs=5,\n",
        "          validation_data=(x_test, y_test))"
      ]
    },
    {
      "cell_type": "code",
      "execution_count": null,
      "metadata": {
        "id": "pPId8Wjg4tIn"
      },
      "outputs": [],
      "source": [
        "# Evaluate the model\n",
        "test_loss, test_acc = model.evaluate(x_test, y_test)\n",
        "print(f\"Test accuracy: {test_acc}\")"
      ]
    }
  ],
  "metadata": {
    "colab": {
      "provenance": [],
      "authorship_tag": "ABX9TyMUUv2UoGxoN3EIYK37m+na",
      "include_colab_link": true
    },
    "kernelspec": {
      "display_name": "Python 3",
      "name": "python3"
    },
    "language_info": {
      "name": "python"
    }
  },
  "nbformat": 4,
  "nbformat_minor": 0
}
