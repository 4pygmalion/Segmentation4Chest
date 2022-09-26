import os
import argparse
import numpy as np
import tensorflow as tf

from model import build_model
from post_processing import post_processing


def get_args() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-m", "--model", help="model path", type=str, required=True
    )
    parser.add_argument(
        "-i", "--image", help="image path", type=str, required=True
    )
    parser.add_argument(
        "-o", "--output", help="output path", type=str, required=True
    )

    return parser.parse_args()


def load_model(model_weight_path: str) -> tf.keras.Model:
    if not os.path.exists(model_weight_path):
        raise FileNotFoundError(f"{model_weight_path} not found")

    model = build_model()
    model.load_weights(model_weight_path)
    return model


def load_image(image_path: str) -> tf.Tensor:
    if not os.path.exists(image_path):
        raise FileNotFoundError(f"{image_path} not found")

    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)
    img = tf.image.convert_image_dtype(img, tf.float32)
    img = tf.image.resize(img, (512, 512), method="nearest")
    return img


if __name__ == "__main__":

    ARGS = get_args()
    IMG = load_image(ARGS.image)
    MODEL = load_model(ARGS.model)
    OUTPUT_IMAGE = MODEL.predict(tf.expand_dims(IMG, axis=0))

    post_processing(MODEL, OUTPUT_IMAGE, ARGS.output)
