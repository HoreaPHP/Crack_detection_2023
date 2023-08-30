import tensorflow as tf

from config import IMAGE_SIZE


def unet_model(downstack, upstack, output_channels):
    inputs = tf.keras.layers.Input(shape=[IMAGE_SIZE, IMAGE_SIZE, 3])
    x = inputs

    # Downsampling through the model
    skips = downstack(x)
    for skip in skips:
        print(skip)

    x = skips[-1]
    skips = reversed(skips[:-1])

    # Upsampling and establishing the skip connections
    for up, skip in zip(upstack, skips):
        concat = tf.keras.layers.Concatenate()

        x = up(x)
        x = concat([x, skip])

    # This is the last layer of the model
    last = tf.keras.layers.Conv2DTranspose(
        output_channels,
        3,
        strides=2,
        padding='same',
        activation='softmax')  #64x64 -> 128x128

    x = last(x)

    return tf.keras.Model(inputs=inputs, outputs=x)
