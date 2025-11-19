import os

import numpy as np
import sklearn.model_selection  # type: ignore
import tensorflow as tf  # type: ignore
from tensorflow.keras import Model, layers, models  # type: ignore


def build_model() -> Model:
    """CNNモデルの構築

    Returns:
        Model: 構築したモデル
    """
    model = models.Sequential(
        [
            layers.Conv2D(
                32,
                (3, 3),
                activation="relu",
                padding="same",
                input_shape=(28, 28, 1),
                name="conv1",
            ),
            layers.MaxPooling2D((2, 2), name="pool1"),
            layers.Conv2D(64, (3, 3), activation="relu", padding="same", name="conv2"),
            layers.MaxPooling2D((2, 2), name="pool2"),
            layers.Flatten(name="flatten"),
            layers.Dense(64, activation="relu", name="dense1"),
            layers.Dense(10, activation="softmax", name="output"),
        ]
    )
    model.compile(
        optimizer="adam", loss="sparse_categorical_crossentropy", metrics=["accuracy"]
    )
    return model


def load_or_train_model(
    model_path: str,
    epochs: int = 10,
    batch_size: int = 128,
    validation_split: float = 0.1,
    augment: bool = True,
) -> Model:
    """モデルを読み込み、なければ学習して保存する

    Args:
        model_path (str): モデルの保存パス
        epochs (int): 学習エポック数
        batch_size (int): バッチサイズ
        validation_split (float): 検証データの割合
        augment (bool): データ拡張を行うかどうか

    Returns:
        Model: 読み込んだまたは学習したモデル
    """
    if os.path.exists(model_path):
        model = tf.keras.models.load_model(model_path)
        print("✅ 既存モデルを読み込みました。再学習をスキップします。")
        return model

    print("🧠 モデルを新規学習中...")
    (x_train, y_train), _ = tf.keras.datasets.mnist.load_data()
    x_train = x_train[..., np.newaxis] / 255.0
    x_train, x_val, y_train, y_val = sklearn.model_selection.train_test_split(
        x_train, y_train, test_size=validation_split, random_state=42
    )

    model = build_model()
    if augment:
        datagen = tf.keras.preprocessing.image.ImageDataGenerator(
            rotation_range=10,
            width_shift_range=0.1,
            height_shift_range=0.1,
            zoom_range=0.1,
        )
        train_gen = datagen.flow(x_train, y_train, batch_size=batch_size)
        steps = x_train.shape[0] // batch_size
        model.fit(
            train_gen,
            epochs=epochs,
            steps_per_epoch=steps,
            validation_data=(x_val, y_val),
            verbose=1,
        )
    else:
        model.fit(
            x_train,
            y_train,
            epochs=epochs,
            batch_size=batch_size,
            validation_data=(x_val, y_val),
            verbose=1,
        )
    model.save(model_path)
    print(f"💾 モデルを保存しました: {model_path}")
    return model
