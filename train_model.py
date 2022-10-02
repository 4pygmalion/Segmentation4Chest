import os
import argparse
import tensorflow as tf
import matplotlib.pyplot as plt
from model import build_model
from tensorflow.keras.callbacks import (
    ModelCheckpoint,
    TensorBoard,
    ReduceLROnPlateau,
)


def get_arg() -> argparse.Namespace:
    parser = argparse.ArgumentParser()
    parser.add_argument(
        "-i", "--image_dir", type=str, help="image folder path", required=True
    )
    parser.add_argument(
        "-v",
        "--val_dir",
        type=str,
        help="image folder path of validation",
        required=True,
    )
    parser.add_argument(
        "-m",
        "--output_model_path",
        type=str,
        help="output model path",
        required=True,
    )
    parser.add_argument("--buffer_size", type=int, help="buffer size")
    parser.add_argument("--batch_size", type=int, help="batch size")
    parser.add_argument("--epochs", type=int, help="epochs")
    return parser.parse_args()


def process_path(image_path, mask_path):
    img = tf.io.read_file(image_path)
    img = tf.image.decode_png(img, channels=3)

    mask = tf.io.read_file(mask_path)
    mask = tf.image.decode_png(mask, channels=1)
    return img, mask


def get_images_mask_path(data_dir: str) -> tuple:
    """Get image and mask data path

    Args:
        data_dir (str): data folder path

    Returns:
        tuple
    """
    image_dir = os.path.join(data_dir, "image")
    image_list = sorted(
        [os.path.join(image_dir, fname) for fname in os.listdir(image_dir)]
    )

    mask_dir = os.path.join(data_dir, "mask")
    mask_list = sorted(
        [os.path.join(mask_dir, fname) for fname in os.listdir(mask_dir)]
    )

    return image_list, mask_list


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def display(display_list):
    plt.figure(figsize=(15, 15))

    title = ["Input Image", "True Mask", "Predicted Mask", "Post Processed"]

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i + 1)
        plt.title(title[i])
        plt.imshow((display_list[i]))
        plt.axis("off")
    plt.show()


if __name__ == "__main__":
    ARGS = get_arg()
    IMG_PATHS, MASK_PATHS = get_images_mask_path(ARGS.image_dir)

    MODEL = build_model()
    MODEL.compile(
        optimizer="adam",
        loss=tf.keras.losses.SparseCategoricalCrossentropy(from_logits=True),
        metrics=["accuracy"],
    )
    dataset = tf.data.Dataset.from_tensor_slices((IMG_PATHS, MASK_PATHS))
    train_ds = dataset.map(process_path)

    CALLBAKCS = [
        ModelCheckpoint(
            filepath=f"{ARGS.output_model_path}"
            + "-{epoch:03d}-{loss:.4f}-{accuracy:.4f}-{val_loss:.4f}-{val_accuracy:.4f}.hdf5",
            monitor="val_loss",
            verbose=1,
            save_best_only=True,
            mode="min",
        ),
        TensorBoard(log_dir="logs", write_graph=True, write_images=True),
        ReduceLROnPlateau(
            monitor="val_loss", factor=0.9, patience=2, verbose=1, min_lr=1e-8
        ),
    ]

    train_dataset = (
        train_ds.cache().shuffle(ARGS.buffer_size).batch(ARGS.batch_size)
    )

    model_history = MODEL.fit(
        train_dataset,
        epochs=ARGS.epochs,
        validation_data=ARGS.val_dir,
        callbacks=CALLBAKCS,
    )
