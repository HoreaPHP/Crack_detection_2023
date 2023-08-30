import cv2
import numpy as np
import os
import matplotlib.pyplot as plt
from PIL import Image

from config import CLASS_MAPPER, IMAGE_SIZE, OG_MASKS_DIR, IMAGES_DIR, MASKS_DIR


def mask_unifier():
    images_names = os.listdir(IMAGES_DIR)

    multi_mask_names = os.listdir(OG_MASKS_DIR)
    # print(multi_mask_names)

    masks_by_image = []
    for image_name in images_names:
        img_name = ".".join(image_name.split(".")[:-1])
        #res = [i for i in multi_mask_names if img_name in i and not i.__contains__("mask")]
        res = [i for i in multi_mask_names if img_name in i and i.__contains__("_z")]
        # print(res)
        masks_by_image.append(res)

    for image_name, masks in zip(images_names, masks_by_image):
        img_name = ".".join(image_name.split(".")[:-1])
        combined_mask = np.zeros((IMAGE_SIZE, IMAGE_SIZE), np.uint8)
        for id, mask_name in enumerate(masks):
            mask_type = ".".join(mask_name.split(".")[:-1]).split('_')[-1]
            mask = cv2.imread(OG_MASKS_DIR + mask_name, cv2.IMREAD_GRAYSCALE)
            mask = cv2.resize(mask, (IMAGE_SIZE, IMAGE_SIZE))

            combined_mask[mask != 0] = CLASS_MAPPER[mask_type]

        print(np.unique(combined_mask, return_counts=True))
        Image.fromarray(combined_mask).save(MASKS_DIR + img_name + '_combined.png')

    print("VALUE CHECK")
    # VALUE CHECK
    for image_name in images_names:
        img_name = ".".join(image_name.split(".")[:-1])
        mask = cv2.imread(MASKS_DIR + f"{img_name}_combined.png", cv2.IMREAD_GRAYSCALE)

        print(np.unique(mask, return_counts=True))


