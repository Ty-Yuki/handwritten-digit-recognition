import base64
import io

import matplotlib.pyplot as plt
import numpy as np
from matplotlib.patches import Rectangle
from tensorflow.keras import Model  # type: ignore


def visualize_activations(
    img28: np.ndarray, model: Model
) -> tuple[int, list[dict[str, str]]]:
    """28x28グレースケール画像をCNNに通して中間層ごとにbase64画像リストを返す

    Args:
        img28 (np.ndarray): 28x28のグレースケール画像配列
        model (Model): 学習済みCNNモデル

    Returns:
        tuple[int, list[dict[str, str]]]: 予測ラベルと中間層画像リスト
    """
    img_arr = img28.astype("float32") / 255.0
    img_arr = np.expand_dims(img_arr, (0, -1))

    layer_names = [
        l.name
        for l in model.layers
        if any(key in l.name for key in ("conv", "pool", "dense", "output"))
    ]
    layer_outputs = [model.get_layer(name).output for name in layer_names]
    activation_model = Model(inputs=model.layers[0].input, outputs=layer_outputs)
    activations = activation_model.predict(img_arr)

    images = []
    for act in activations:
        if act.ndim == 2:
            plt.figure(figsize=(0.4, 2))
            fmap = act[0].reshape(-1, 1)
            fmap = (fmap - fmap.min()) / (fmap.max() + 1e-6)
            plt.imshow(fmap, cmap="gray", aspect="auto")
            num_units = fmap.shape[0]
            for y in range(num_units):
                if num_units <= 10:
                    plt.text(
                        1.0,
                        y,
                        str(y),
                        color="black",
                        fontsize=8,
                        va="center",
                        ha="center",
                    )
                if y < num_units - 1:
                    plt.axhline(y + 0.5, color="white", linewidth=1.0)
            plt.axhline(-0.5, color="black", linewidth=1.0)
            plt.axhline(num_units - 0.5, color="black", linewidth=2.0)
            plt.axvline(-0.5, color="black", linewidth=1.0)
            plt.axvline(0.5, color="black", linewidth=2.0)
        else:
            plt.figure(figsize=(2, 2))
            fmap = act[0, :, :, 0]
            fmap = (fmap - fmap.min()) / (fmap.max() + 1e-6)
            plt.imshow(fmap, cmap="gray", extent=(0, fmap.shape[1], fmap.shape[0], 0))
            ax = plt.gca()
            rect = Rectangle(
                (0, 0),
                fmap.shape[1],
                fmap.shape[0],
                linewidth=1,
                edgecolor="black",
                facecolor="none",
            )
            ax.add_patch(rect)
        plt.axis("off")
        buf = io.BytesIO()
        plt.savefig(buf, format="png", bbox_inches="tight", pad_inches=0)
        plt.close()
        buf.seek(0)
        img_base64 = base64.b64encode(buf.getvalue()).decode("utf-8")
        images.append({"img": img_base64})

    pred = int(np.argmax(model.predict(img_arr), axis=1)[0])
    return pred, images
