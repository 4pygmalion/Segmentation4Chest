import os
import glob
import numpy as np
from PIL import Image

import Augmentor

ROOT_DIR = os.path.dirname(__file__)
DATA_DIR = os.path.join(ROOT_DIR,"data")
TRAIN_DIR = os.path.join(DATA_DIR, "train")
TEST_DIR = os.path.join(DATA_DIR, "test")

if __name__ == "__main__":
    
    # option 1
    ground_truth_images = glob.glob(os.path.join(TRAIN_DIR, "img", "*"))
    segmentation_mask_images = glob.glob(os.path.join(TRAIN_DIR, "mask", "*"))
    collated_images_and_masks = list(zip(ground_truth_images, segmentation_mask_images))

    images = [[np.asarray(Image.open(x)), np.asarray(Image.open(y))] for x, y in collated_images_and_masks]

    AugmentationPipeline = Augmentor.DataPipeline(images)
    AugmentationPipeline.random_distortion(probability=1, grid_width=5, grid_height=5, magnitude=8)
    AugmentationPipeline.gaussian_distortion(probability=1, grid_width=5, grid_height=5, magnitude=8, corner='bell', method='in ')
    augmented_images = AugmentationPipeline.sample(1000)

    cnt = 1
    for img, mask in augmented_images:
        imag = Image.fromarray(img)
        imag.save(os.path.join(TRAIN_DIR, "img", f"aug_{cnt}.png"))
        mask = Image.fromarray(mask)
        mask.save(os.path.join(TRAIN_DIR, "mask", f"aug_{cnt}.png"))
        cnt += 1

    # option 2
    AugmentationPipeline = Augmentor.DataPipeline(images)
    AugmentationPipeline.random_distortion(probability=1, grid_width=7, grid_height=7, magnitude=10)
    AugmentationPipeline.gaussian_distortion(probability=1, grid_width=7, grid_height=7, magnitude=10, corner='bell', method='in ')
    augmented_images = AugmentationPipeline.sample(500)

    for img, mask in augmented_images:
        imag = Image.fromarray(img)
        imag.save(os.path.join(TRAIN_DIR, "img", f"aug_{cnt}.png"))
        mask = Image.fromarray(mask)
        mask.save(os.path.join(TRAIN_DIR, "mask", f"aug_{cnt}.png"))
        cnt += 1