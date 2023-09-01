# Crack_detection_2023

Required lybraries:
sklearn, tensorflow, IPython, opencv-python, numpy, imgaug, matplotlib, keras, Pillow

You also require the pix2pix library for the upsampling path. You need to do the following:
`git clone https://github.com/tensorflow/examples.git`

Then go to the examples folder an run the following command:
`pip install -e .` this allow editing of the library

After you have installed the library go to the `pix2pix.py` and change the upsampling path method with the following:

```
def upsample(filters, size, strides, norm_type='batchnorm', apply_dropout=False):
  """Upsamples an input.

  Conv2DTranspose => Batchnorm => Dropout => Relu

  Args:
    filters: number of filters
    size: filter size
    norm_type: Normalization type; either 'batchnorm' or 'instancenorm'.
    apply_dropout: If True, adds the dropout layer

  Returns:
    Upsample Sequential Model
  """

  initializer = tf.random_normal_initializer(0., 0.02)

  result = tf.keras.Sequential()
  result.add(
      tf.keras.layers.Conv2DTranspose(filters, size, strides=strides,
                                      padding='same',
                                      kernel_initializer=initializer,
                                      use_bias=False))

  if norm_type.lower() == 'batchnorm':
    result.add(tf.keras.layers.BatchNormalization())
  elif norm_type.lower() == 'instancenorm':
    result.add(InstanceNormalization())

  if apply_dropout:
    result.add(tf.keras.layers.Dropout(0.5))

  result.add(tf.keras.layers.ReLU())

  return result
```

It is higly recommended to run the code on the GPU since it is quite complex and it requires a long time to train
In order to run the code you can just run "python train.py".

The whole process is automated. If you want to combine the masks in a different way you can modify line 20 in `mask_unifier.py`
