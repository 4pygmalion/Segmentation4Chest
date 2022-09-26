import cv2
import numpy as np
import tensorflow as tf


def create_mask(pred_mask):
    pred_mask = tf.math.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def do_floodFIll(image):
    im_in = image
    th, im_th = cv2.threshold(im_in, 0, 1, cv2.THRESH_BINARY_INV)
    im_floodfill = im_th.copy()
    h, w = im_th.shape[:2]
    mask = np.zeros((h + 2, w + 2), np.uint8)

    retval, _image, _mask, _rect = cv2.floodFill(im_floodfill, mask, (3, 3), 0)

    resized_fill_mask = cv2.resize(
        _mask, dsize=(512, 512), interpolation=cv2.INTER_CUBIC
    )

    res_mask = resized_fill_mask.copy()
    res_mask[np.where(res_mask == 1)] = 9
    res_mask[np.where(res_mask == 0)] = 1
    res_mask[np.where(res_mask == 9)] = 0

    return res_mask


def overlay_cardiac_mask(original_mask, cardiac_mask):
    new_mask = np.zeros(shape=(512, 512))

    new_mask[np.where(original_mask == 2)] = 2
    new_mask[np.where(cardiac_mask == 1)] = 1
    return new_mask


def convert_to_3channel(image, original_size):

    new_image = np.zeros(shape=(*original_size, 3), dtype=np.uint8)

    resize_original_image = cv2.resize(
        image, dsize=original_size, interpolation=cv2.INTER_CUBIC
    )
    new_image[:, :, 0][np.where(resize_original_image == 0)] = 1
    new_image[:, :, 1][np.where(resize_original_image == 1)] = 1
    new_image[:, :, 2][np.where(resize_original_image == 2)] = 1
    return new_image


def post_processing(
    model, output_image: tf.Tensor, output_path, original_size=(512, 512)
):

    labeled_mask = create_mask(output_image)

    labeled_mask_array = labeled_mask.numpy().reshape(512, 512)
    labeled_mask_array = labeled_mask_array.astype(np.uint8)

    opened_mask_array = cv2.dilate(
        labeled_mask_array, cv2.getStructuringElement(cv2.MORPH_RECT, (10, 10))
    )
    closing_mask_array = cv2.morphologyEx(
        opened_mask_array,
        cv2.MORPH_CLOSE,
        cv2.getStructuringElement(cv2.MORPH_RECT, (5, 5)),
    )
    opened_mask_array = cv2.morphologyEx(
        closing_mask_array, cv2.MORPH_OPEN, np.ones((17, 17), np.uint8)
    )

    fill_cardiac = do_floodFIll(labeled_mask_array)
    final_mask = overlay_cardiac_mask(opened_mask_array, fill_cardiac)

    new_image = convert_to_3channel(final_mask, original_size=original_size)
    np.save(output_path, new_image)
