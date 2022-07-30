import os
import argparse
import pydicom as dicom
import matplotlib.pyplot as plt
import cv2

def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("-s", "--src_dir", type=str, help="source directory")
    parser.add_argument("-d", "--dst_dir", type=str, help="destination directory")

    return parser.parse_args()

def dicom2png(src:str, dst:str) -> None:
    dicom_ojb = dicom.dcmread(src)
    plt.imsave(dst, cv2.normalize(dicom_ojb.pixel_array, None, 0, 255, cv2.NORM_MINMAX))

    return 


if __name__ == "__main__":
    
    ARGS = get_args()
    DATA_DIR = ARGS.src_dir
    ROOT_DIR = os.path.dirname(DATA_DIR)
    DST_DIR = ARGS.dst_dir

    os.makedirs(DST_DIR, exist_ok=True)

    for dcm in os.listdir(DATA_DIR):
        src = os.path.join(DATA_DIR, dcm)
        dst = os.path.join(DST_DIR, dcm.replace(".dcm", ".png"))
        dicom2png(src, dst)
