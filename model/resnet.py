from keras.applications.resnet50 import ResNet50

# -*- coding: utf-8 -*-
"""ResNet50 model for Keras.

# Reference:

- [Deep Residual Learning for Image Recognition](https://arxiv.org/abs/1512.03385)

Adapted from code contributed by BigMoyan.
"""

from keras.layers import Input
from keras import layers
from keras.layers import Activation
from keras.layers import Conv2D
from keras.layers import MaxPooling2D
from keras.layers import BatchNormalization, UpSampling2D, concatenate
from keras.models import Model
from keras import backend as K
from keras.optimizers import SGD
from keras.utils.data_utils import get_file
from keras.applications.imagenet_utils import _obtain_input_shape
from keras.utils.vis_utils import plot_model
from model.u_net import relu, leaky
from model.losses import bce_dice_loss, dice_coeff



WEIGHTS_PATH = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels.h5'
WEIGHTS_PATH_NO_TOP = 'https://github.com/fchollet/deep-learning-models/releases/download/v0.2/resnet50_weights_tf_dim_ordering_tf_kernels_notop.h5'


def identity_block(input_tensor, kernel_size, filters, stage, block):
    """The identity block is the block that has no conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'keras.., current block label, used for generating layer names

    # Returns
        Output tensor for the block.
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), name=conv_name_base + '2a', trainable=False)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', trainable=False)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size,
               padding='same', name=conv_name_base + '2b', trainable=False)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', trainable=False)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', trainable=False)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', trainable=False)(x)

    x = layers.add([x, input_tensor])
    x = Activation('relu')(x)
    return x


def conv_block(input_tensor, kernel_size, filters, stage, block, strides=(2, 2)):
    """A block that has a conv layer at shortcut.

    # Arguments
        input_tensor: input tensor
        kernel_size: default 3, the kernel size of middle conv layer at main path
        filters: list of integers, the filterss of 3 conv layer at main path
        stage: integer, current stage label, used for generating layer names
        block: 'a','b'keras.., current block label, used for generating layer names

    # Returns
        Output tensor for the block.

    Note that from stage 3, the first conv layer at main path is with strides=(2,2)
    And the shortcut should have strides=(2,2) as well
    """
    filters1, filters2, filters3 = filters
    if K.image_data_format() == 'channels_last':
        bn_axis = 3
    else:
        bn_axis = 1
    conv_name_base = 'res' + str(stage) + block + '_branch'
    bn_name_base = 'bn' + str(stage) + block + '_branch'

    x = Conv2D(filters1, (1, 1), strides=strides,
               name=conv_name_base + '2a', trainable=False)(input_tensor)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2a', trainable=False)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters2, kernel_size, padding='same',
               name=conv_name_base + '2b', trainable=False)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2b', trainable=False)(x)
    x = Activation('relu')(x)

    x = Conv2D(filters3, (1, 1), name=conv_name_base + '2c', trainable=False)(x)
    x = BatchNormalization(axis=bn_axis, name=bn_name_base + '2c', trainable=False)(x)

    shortcut = Conv2D(filters3, (1, 1), strides=strides,
                      name=conv_name_base + '1', trainable=False)(input_tensor)
    shortcut = BatchNormalization(axis=bn_axis, name=bn_name_base + '1', trainable=False)(shortcut)

    x = layers.add([x, shortcut])
    x = Activation('relu')(x)
    return x


def ResNet50(input_shape=None,
              num_classes=1,
             optimizer=SGD(),
             activation=relu,
             regularizer=None):
    """Instantiates the ResNet50 architecture.

    Optionally loads weights pre-trained
    on ImageNet. Note that when using TensorFlow,
    for best performance you should set
    `image_data_format="channels_last"` in your Keras config
    at ~/.keras/keras.json.

    The model and the weights are compatible with both
    TensorFlow and Theano. The data format
    convention used by the model is the one
    specified in your Keras config file.

    # Arguments
        include_top: whether to include the fully-connected
            layer at the top of the network.
        weights: one of `None` (random initialization)
            or "imagenet" (pre-training on ImageNet).
        input_tensor: optional Keras tensor (i.e. output of `layers.Input()`)
            to use as image input for the model.
        input_shape: optional shape tuple, only to be specified
            if `include_top` is False (otherwise the input shape
            has to be `(224, 224, 3)` (with `channels_last` data format)
            or `(3, 224, 224)` (with `channels_first` data format).
            It should have exactly 3 inputs channels,
            and width and height should be no smaller than 197.
            E.g. `(200, 200, 3)` would be one valid value.
        pooling: Optional pooling mode for feature extraction
            when `include_top` is `False`.
            - `None` means that the output of the model will be
                the 4D tensor output of the
                last convolutional layer.
            - `avg` means that global average pooling
                will be applied to the output of the
                last convolutional layer, and thus
                the output of the model will be a 2D tensor.
            - `max` means that global max pooling will
                be applied.
        classes: optional number of classes to classify images
            into, only to be specified if `include_top` is True, and
            if no `weights` argument is specified.

    # Returns
        A Keras model instance.

    # Raises
        ValueError: in case of invalid argument for `weights`,
            or invalid input shape.
    """

    img_input = Input(shape=input_shape)

    bn_axis = 3

    # 1x
    x1bn = BatchNormalization()(img_input)
    x1 = Conv2D(64, (7, 7), strides=(2, 2), name='conv1', padding='same', trainable=False)(x1bn)

    # 2x
    x2 = BatchNormalization(axis=bn_axis, name='bn_conv1', trainable=False)(x1)
    x2 = Activation('relu')(x2)
    x2 = MaxPooling2D((2, 2), strides=(2, 2))(x2)

    # 4x
    x3 = conv_block(x2, 3, [64, 64, 256], stage=2, block='a', strides=(1, 1))
    x3 = identity_block(x3, 3, [64, 64, 256], stage=2, block='b')
    x3 = identity_block(x3, 3, [64, 64, 256], stage=2, block='c')
    x3 = conv_block(x3, 3, [128, 128, 512], stage=3, block='a')

    # 8x
    x4 = identity_block(x3, 3, [128, 128, 512], stage=3, block='b')
    x4 = identity_block(x4, 3, [128, 128, 512], stage=3, block='c')
    x4 = identity_block(x4, 3, [128, 128, 512], stage=3, block='d')
    x4= conv_block(x4, 3, [256, 256, 1024], stage=4, block='a')

    # 16x
    x5 = identity_block(x4, 3, [256, 256, 1024], stage=4, block='b')
    x5 = identity_block(x5, 3, [256, 256, 1024], stage=4, block='c')
    x5 = identity_block(x5, 3, [256, 256, 1024], stage=4, block='d')
    x5 = identity_block(x5, 3, [256, 256, 1024], stage=4, block='e')
    x5 = identity_block(x5, 3, [256, 256, 1024], stage=4, block='f')
    x5 = conv_block(x5, 3, [512, 512, 2048], stage=5, block='a')

    # 32x
    x6 = identity_block(x5, 3, [512, 512, 2048], stage=5, block='b')
    x6 = identity_block(x6, 3, [512, 512, 2048], stage=5, block='c')
    x6 = UpSampling2D()(x6)

    # 16x
    x7 = concatenate([x6, x4])
    x7 = Conv2D(256, (1,1), padding='same', kernel_regularizer=regularizer, name='b1_conv1')(x7)
    x7 = BatchNormalization(name='b1_bn1')(x7)
    x7 = activation()(x7)
    x7 = Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizer, name='b1_conv2')(x7)
    x7 = BatchNormalization(name='b1_bn2')(x7)
    x7 = activation()(x7)
    x7 = Conv2D(256, (3,3), padding='same', kernel_regularizer=regularizer, name='b1_conv3')(x7)
    x7 = BatchNormalization(name='b1_bn3')(x7)
    x7 = activation()(x7)
    x7 = UpSampling2D()(x7)

    # 8x
    x8 = concatenate([x7, x3])
    x8 = Conv2D(128, (1,1), padding='same', kernel_regularizer=regularizer, name='b2_conv1')(x8)
    x8 = BatchNormalization(name='b2_bn1')(x8)
    x8 = activation()(x8)
    x8 = Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizer, name='b2_conv2')(x8)
    x8 = BatchNormalization(name='b2_bn2')(x8)
    x8 = activation()(x8)
    x8 = Conv2D(128, (3,3), padding='same', kernel_regularizer=regularizer, name='b2_conv3')(x8)
    x8 = BatchNormalization(name='b2_bn3')(x8)
    x8 = activation()(x8)
    x8 = UpSampling2D()(x8)

    # 4x
    x9 = concatenate([x8, x2])
    x9 = Conv2D(64, (1,1), padding='same', kernel_regularizer=regularizer, name='b3_conv1')(x9)
    x9 = BatchNormalization(name='b3_bn1')(x9)
    x9 = activation()(x9)
    x9 = Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizer, name='b3_conv2')(x9)
    x9 = BatchNormalization(name='b3_bn2')(x9)
    x9 = activation()(x9)
    x9 = Conv2D(64, (3,3), padding='same', kernel_regularizer=regularizer, name='b3_conv3')(x9)
    x9 = BatchNormalization(name='b3_bn3')(x9)
    x9 = activation()(x9)
    x9 = UpSampling2D()(x9)

    # 2x
    x10 = concatenate([x9, x1])
    x10 = Conv2D(32, (1,1), padding='same', kernel_regularizer=regularizer, name='b4_conv1')(x10)
    x10 = BatchNormalization(name='b4_bn1')(x10)
    x10 = activation()(x10)
    x10 = Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizer, name='b4_conv2')(x10)
    x10 = BatchNormalization(name='b4_bn2')(x10)
    x10 = activation()(x10)
    x10 = Conv2D(32, (3,3), padding='same', kernel_regularizer=regularizer, name='b4_conv3')(x10)
    x10 = BatchNormalization(name='b4_bn3')(x10)
    x10 = activation()(x10)
    x10 = UpSampling2D()(x10)

    # x1
    x11 = concatenate([x10, x1bn])
    x11 = Conv2D(16, (1,1), padding='same', kernel_regularizer=regularizer, name='b5_conv1')(x11)
    x11 = BatchNormalization(name='b5_bn1')(x11)
    x11 = activation()(x11)
    x11 = Conv2D(16, (3,3), padding='same', kernel_regularizer=regularizer, name='b5_conv2')(x11)
    x11 = BatchNormalization(name='b5_bn2')(x11)
    x11 = activation()(x11)
    x11 = Conv2D(16, (3,3), padding='same', kernel_regularizer=regularizer, name='b5_conv3')(x11)
    x11 = BatchNormalization(name='b5_bn3')(x11)
    x11 = activation()(x11)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='classify')(x11)
    model = Model(inputs=img_input, outputs=classify)

    model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coeff])

    return model


if __name__ == '__main__':
    model = ResNet50(input_shape=(256, 256, 3))
    plot_model(model, show_shapes=True)
