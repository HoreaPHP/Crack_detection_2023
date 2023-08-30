import cv2
import numpy as np
import matplotlib.pyplot as plt
import keras


def detect():
    # Load and preprocess the image
    #image_path = '00000002.jpg'
    image_path = "./dataset/images/IMG_4495.JPG"
    mask_path = "./dataset/combined_masks/IMG_4495_combined.png"

    image = cv2.imread(image_path)
    image = cv2.resize(image, (1024, 1024))  # Resize to match model input shape
    # image = image / 255.0  # Normalize pixel values to the range of [0, 1]
    image = np.expand_dims(image, axis=0)  # Add batch dimension

    # Load the trained model
    model = keras.models.load_model("models/mobile_large_1024_only_cracks.h5")

    # Generate predictions
    predictions = model.predict(image)
    # predictions = (predictions > 0.5).astype(np.uint8)  # Threshold the predictions
    predictions = np.argmax(predictions, axis=-1).astype(np.uint8)  # Threshold the predictions

    mask = cv2.imread(mask_path, cv2.IMREAD_GRAYSCALE)

    # Visualize the original image and the predicted mask
    plt.subplot(1, 3, 1)
    plt.imshow(image[0])
    plt.title("Original Image")
    plt.axis('off')

    plt.subplot(1, 3, 2)
    plt.imshow(mask, cmap="gray")
    plt.title("True mask")
    plt.axis('off')

    plt.subplot(1, 3, 3)
    plt.imshow(predictions[0], cmap="gray")
    plt.title("Detected crack")
    plt.axis('off')

    plt.subplots_adjust(wspace=0.5)
    plt.savefig('combined.png')
    plt.imsave('4495.png')
    plt.show()



detect()
