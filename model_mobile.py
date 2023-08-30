import tensorflow as tf
from tensorflow_examples.models.pix2pix import pix2pix

from config import IMAGE_SIZE, CLASSES
# install: pip install -q git+https://github.com/tensorflow/examples.git
from model_creation import unet_model


def mobile_model():
    mobilev2_base_model = tf.keras.applications.MobileNetV2(input_shape=[IMAGE_SIZE, IMAGE_SIZE, 3], include_top=False)

    # Use the activations of these layers - input 208x208
    # mobilev2_layer_names = [
    #     'block_1_expand_relu',  # 64x64
    #     'block_3_expand_relu',  # 32x32
    #     'block_6_expand_relu',  # 16x16
    #     'block_13_expand_relu',  # 8x8
    #     'block_16_project',  # 4x4
    # ]

    mobilev2_layer_names = [
        'block_1_expand_relu',  # 104x104
        'block_2_expand_relu',  # 104x104
        'block_3_expand_relu',  # 52x52
        'block_4_expand_relu',  # 52x52
        'block_5_expand_relu',  # 52x52
        'block_6_expand_relu',  # 26x26
        'block_7_expand_relu',  # 26x26
        'block_8_expand_relu',  # 26x26
        'block_9_expand_relu',  # 26x26
        'block_10_expand_relu',  # 13x13
        'block_11_expand_relu',  # 13x13
        'block_12_expand_relu',  # 13x13
    ]

    mobilev2_layers = [mobilev2_base_model.get_layer(name).output for name in mobilev2_layer_names]
    # print(mobilev2_layers)

    # Create the feature extraction model
    mobilev2_down_stack = tf.keras.Model(inputs=mobilev2_base_model.input, outputs=mobilev2_layers)
    # print(mobilev2_down_stack.summary())
    mobilev2_down_stack.trainable = False

    # mobilev2_up_stack = [
    #     pix2pix.upsample(512, 3),  # 4x4 -> 8x8
    #     pix2pix.upsample(256, 3),  # 8x8 -> 16x16
    #     pix2pix.upsample(128, 3),  # 16x16 -> 32x32
    #     pix2pix.upsample(64, 3),   # 32x32 -> 64x64
    # ]

    mobilev2_up_stack = [
        pix2pix.upsample(1024, 3, strides=1, apply_dropout=0.3),  # 13x13
        pix2pix.upsample(1024, 3, strides=1, apply_dropout=0.3),  # 13x13
        pix2pix.upsample(1024, 3, strides=1, apply_dropout=0.3),  # 13x13
        pix2pix.upsample(1024, 3, strides=1, apply_dropout=0.3),  # 13x13
        pix2pix.upsample(1024, 3, strides=1, apply_dropout=0.3),  # 13x13
        pix2pix.upsample(512, 3, apply_dropout=0.3),  # 26x26
        pix2pix.upsample(512, 3, strides=1, apply_dropout=0.3),  # 26x26
        pix2pix.upsample(512, 3, strides=1, apply_dropout=0.3),  # 26x26
        pix2pix.upsample(256, 3, apply_dropout=0.3),  # 52x52
        pix2pix.upsample(256, 3, strides=1, apply_dropout=0.3),  # 52x52
        pix2pix.upsample(128, 3, apply_dropout=0.3),  # 104x104
        pix2pix.upsample(64, 3, apply_dropout=0.3),  # 208x208
    ]

    model = unet_model(mobilev2_down_stack, mobilev2_up_stack, CLASSES)

    return model