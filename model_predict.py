import matplotlib.pyplot as plt
import tensorflow as tf


def create_mask(pred_mask):
    pred_mask = tf.argmax(pred_mask, axis=-1)
    pred_mask = pred_mask[..., tf.newaxis]
    return pred_mask[0]


def display_data_pred(display_list):
    plt.figure(figsize=(15, 15))

    title = ['Input Image', 'True Mask', 'Predicted Mask']

    for i in range(len(display_list)):
        plt.subplot(1, len(display_list), i+1)
        plt.title(title[i])
        plt.imshow(tf.keras.preprocessing.image.array_to_img(display_list[i]), cmap='jet', interpolation='false')
        plt.axis('off')
    plt.show()


def show_predictions_on_data(model, sample_image, sample_mask):
    display_data_pred([sample_image, sample_mask, create_mask(model.predict(sample_image[tf.newaxis, ...]))])