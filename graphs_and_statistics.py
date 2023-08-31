import keras
import os
import numpy as np
import cv2
from config import *
from sklearn.metrics import confusion_matrix, classification_report, accuracy_score, log_loss
import matplotlib.pyplot as plt
import seaborn as sns

def acc_matrix(masks, predictions):
    # Compute the confusion matrix
    conf_matrix = np.zeros((2, 2))
    for actual, pred in zip(masks, predictions):
        conf_matrix[actual, pred] += 1

    # Convert counts to percentages
    matrix = np.zeros((2, 2))

    total_0 = np.sum(conf_matrix[0, :])
    total_1 = np.sum(conf_matrix[1, :])

    # For class 0
    matrix[0, 0] = conf_matrix[0, 0] / total_0 * 100  # Specificity (True Negative Rate)
    matrix[0, 1] = conf_matrix[0, 1] / total_0 * 100  # False Positive Rate

    # For class 1
    matrix[1, 0] = conf_matrix[1, 0] / total_1 * 100  # False Negative Rate
    matrix[1, 1] = conf_matrix[1, 1] / total_1 * 100  # Sensitivity (True Positive Rate)

    labels = ["Non-cracks", "Cracks"]
    sns.heatmap(matrix, annot=True, cmap='Blues', fmt=".2f", xticklabels=labels, yticklabels=labels)
    plt.ylabel('True labels')
    plt.xlabel('Predicted labels')
    plt.savefig("graphs/accuracy_matrix.png")
    plt.show()

    FP = len(np.where(predictions - masks == 1)[0])
    FN = len(np.where(predictions - masks == -1)[0])
    TP = len(np.where(predictions + masks == 2)[0])
    TN = len(np.where(predictions + masks == 0)[0])
    cmat = [[TP, FN], [FP, TN]]

    plt.figure(figsize = (6,6))
    sns.heatmap(cmat/np.sum(cmat), cmap="Reds", annot=True, fmt = '.2%', square=1,   linewidth=2.)
    plt.xlabel("predictions")
    plt.ylabel("real values")
    plt.show()
    cm = confusion_matrix(masks, predictions)

    total = np.sum(cm)
    cm_percentage = (cm / total) * 100
    cm_percentage_str = np.array([["{:.2f}%".format(val) for val in row] for row in cm_percentage])

    # Using the 's' format specifier since the annotations are now strings
    plt.figure(figsize=(10, 8))
    sns.heatmap(cm_percentage, annot=cm_percentage_str, fmt='s', cmap='Blues')
    plt.xlabel('Predicted')
    plt.ylabel('True')
    plt.savefig("graphs/confusion_matrix.png", dpi=300, bbox_inches='tight')
    plt.show()


def f1_precision_recall(masks, predictions):
    print(classification_report(masks, predictions))


def compute_accuracy(masks, predictions):
    print(accuracy_score(masks, predictions))


def logistic_loss(masks, predictions):
    print(log_loss(masks, predictions))

def plot_3_images(images, masks, predictions):
    plt.subplot(2, 3, 1)
    plt.imshow(images[3])
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(2, 3, 2)
    plt.imshow(masks[3], cmap="gray")
    plt.title("True mask")
    plt.axis('off')

    plt.subplot(2, 3, 3)
    plt.imshow(predictions[3], cmap="gray")
    plt.title("Detected crack")
    plt.axis('off')

    plt.subplot(2, 3, 4)
    plt.imshow(images[1])
    plt.axis('off')

    plt.subplot(2, 3, 5)
    plt.imshow(masks[1], cmap="gray")
    plt.axis('off')

    plt.subplot(2, 3, 6)
    plt.imshow(predictions[1], cmap="gray")
    plt.axis('off')


    plt.subplots_adjust(wspace=0.2)
    plt.savefig('combined.png')
    plt.show()

model = keras.models.load_model('models/mobile_large_1024_only_cracks.h5')

test_images = os.listdir(IMAGES_DIR)
test_masks = os.listdir(MASKS_DIR)

# print(test_images)
# print(test_masks)

predictions = []
images = []
for image in test_images:
    img = cv2.imread(IMAGES_DIR + image)
    img = cv2.resize(img, (IMAGE_SIZE, IMAGE_SIZE))
    images.append(img)
    img = np.expand_dims(img, axis=0)
    predicted_image = model.predict(img)
    predicted_image = np.argmax(predicted_image, axis=-1).astype(np.uint8)
    predictions.append(predicted_image[0])

masks = []
for mask in test_masks:
    mask_img = cv2.imread(MASKS_DIR + mask, cv2.IMREAD_GRAYSCALE)
    masks.append(mask_img)

flattened_predictions = [pred.flatten() for pred in predictions]
flattened_masks = [mask.flatten() for mask in masks]

# Concatenate for confusion calculations
concatenated_predictions = np.concatenate(flattened_predictions)
concatenated_masks = np.concatenate(flattened_masks)

acc_matrix(concatenated_masks, concatenated_predictions)
f1_precision_recall(concatenated_masks, concatenated_predictions)
compute_accuracy(concatenated_masks, concatenated_predictions)
logistic_loss(concatenated_masks, concatenated_predictions)

plot_3_images(images, masks, predictions)


