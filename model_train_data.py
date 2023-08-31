import os

import imgaug.augmenters as iaa
from imgaug.augmentables.segmaps import SegmentationMapsOnImage
import tensorflow as tf
import cv2
import numpy as np

from config import CLASSES_WEIGHTS, CLASSES_WEIGHTS_SUM, IMAGE_SIZE, MASKS_DIR, IMAGES_DIR, BATCH_SIZE


def add_sample_weights(image, label):
    # The weights for each class, with the constraint that:
    class_weights = tf.constant(CLASSES_WEIGHTS)

    class_weights = class_weights/tf.reduce_sum(class_weights)

    # Create an image of `sample_weights` by using the label at each pixel as an
    # index into the `class weights` .
    sample_weights = tf.gather(class_weights, indices=tf.cast(label, tf.int32))

    return image, label, sample_weights


# Define our augmentation pipeline.
seq = iaa.Sequential([
    iaa.Fliplr(0.5), # horizontally flip 50% of the images
    iaa.Crop(percent=(0, 0.05)),  # random crops
    # # Small gaussian blur with random sigma between 0 and 0.5.
    # # But we only blur about 50% of all images.
    # Add a value of -10 to 10 to each pixel.
    iaa.Add((-10, 10)),
    iaa.Sometimes(
        0.1,
        iaa.Dropout([0.01, 0.05]),  # drop 5% or 20% of all pixels
    ),
    # iaa.Sometimes(
    #     0.1,
    #     iaa.GaussianBlur(sigma=(0, 0.1))
    # ),
    iaa.Sometimes(
        0.25,
        iaa.Affine(rotate=(-45, 45)),  # rotate by -45 to 45 degrees (affects segmaps)
        iaa.Sharpen((0.0, 1.0)),  # sharpen the image
    ),
    # # Strengthen or weaken the contrast in each image.
    iaa.Sometimes(
        0.1,
        iaa.LinearContrast((0.95, 1.05)),
    ),
    # iaa.LinearContrast((0.95, 1.05)),
    # iaa.ElasticTransformation(alpha=50, sigma=5)  # apply water effect (affects segmaps)
], random_order=True)


class MyIterator:
    def __init__(self, images, masks, BATCH_SIZE):
        self.images = images
        self.masks = masks
        self.BATCH_SIZE = BATCH_SIZE

    def __iter__(self):
        return self

    def __next__(self):
        random_indexes = np.random.randint(0,len(self.images), self.BATCH_SIZE)
        images_batch = []
        masks_batch = []
        weights_batch = []
        for index in random_indexes:
            image, mask, weights = add_sample_weights(self.images[index], self.masks[index])

            zeros = np.zeros_like(mask)
            weights = weights * CLASSES_WEIGHTS_SUM
            merged_mask_weights = cv2.merge((np.uint8(mask), np.uint8(weights), np.uint8(zeros)))

            mask = SegmentationMapsOnImage(merged_mask_weights, shape=image.shape)

            image = image.astype(np.uint8)
            aug_image, aug_mask_weights = seq(image=image, segmentation_maps=mask)

            aug_mask, aug_weights, _ = cv2.split(aug_mask_weights.get_arr())

            aug_weights = np.array(aug_weights, dtype=np.int64)
            aug_mask = aug_mask[:, :, None]
            aug_weights = aug_weights[:, :, None]

            aug_weights = aug_weights / CLASSES_WEIGHTS_SUM


            # display([np.uint8(aug_image), np.uint8(aug_mask), aug_weights/CLASSES_WEIGHTS_SUM])

            images_batch.append(aug_image)
            masks_batch.append(aug_mask)
            weights_batch.append(aug_weights)

        return np.array(images_batch), np.array(masks_batch), np.array(weights_batch)


def load_images_and_masks():
    images = []
    masks = []
    for filename in os.listdir(IMAGES_DIR):
        image_path = os.path.join(IMAGES_DIR, filename)

        image = cv2.imread(image_path)
        image = cv2.resize(image, (IMAGE_SIZE, IMAGE_SIZE))
        image = image.astype(np.uint8)

        images.append(image)

    for filename in os.listdir(MASKS_DIR):
        if filename.endswith('.png'):
            mask_path = os.path.join(MASKS_DIR, filename)
            mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)
            masks.append(mask)

    images = np.array(images)
    masks = np.array(masks)
    masks = np.expand_dims(masks, axis=-1)
    print(images.shape)
    print(masks.shape)

    return images, masks



def model_data_generator(show=False):
    images, masks = load_images_and_masks()

    #display([masks[0]])
    #display([images[0], masks[0]])


    trainGeneratorInstance = MyIterator(images, masks, BATCH_SIZE)
    trainGenerator = iter(trainGeneratorInstance)

    valGeneratorInstance = MyIterator(images, masks, BATCH_SIZE)
    valGenerator = iter(valGeneratorInstance)


    images_batch, masks_batch, weights_batch = trainGenerator.__next__()
    for i in range(BATCH_SIZE):
        sample_image = images_batch[i]
        sample_mask = masks_batch[i]
        sample_weights = weights_batch[i]
        #print(np.unique(sample_mask, return_counts=True))
        #print(np.unique(sample_weights, return_counts=True))
        # display([sample_mask])
        break

    return trainGenerator, valGenerator, (sample_image, sample_mask, sample_weights)

