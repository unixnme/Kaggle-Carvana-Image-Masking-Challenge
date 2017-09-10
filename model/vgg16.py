import keras
from keras.layers import Input, Conv2D, MaxPooling2D, UpSampling2D, BatchNormalization, Activation
from keras.layers.merge import Concatenate
from keras.optimizers import SGD
from keras.models import Model
from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff
from model.u_net import leaky

num_classes = 1

def vgg16(input_shape=(128, 128, 3), regularizer=None, optimizer=SGD, activation=leaky):
    img_input = Input(shape=input_shape)

    # Block 1
    x1 = Conv2D(64, (3, 3), trainable=False, activation='relu', padding='same', name='block1_conv1')(img_input)
    x1 = Conv2D(64, (3, 3), trainable=False, activation='relu', padding='same', name='block1_conv2')(x1)
    # 64, 224, 224
    x1_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x1)
    # 64, 112, 112

    # Block 2
    x2 = Conv2D(128, (3, 3), trainable=False, activation='relu', padding='same', name='block2_conv1')(x1_pool)
    x2 = Conv2D(128, (3, 3), trainable=False, activation='relu', padding='same', name='block2_conv2')(x2)
    x2_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x2)
    # 128, 56, 56

    # Block 3
    x3 = Conv2D(256, (3, 3), trainable=False, activation='relu', padding='same', name='block3_conv1')(x2_pool)
    x3 = Conv2D(256, (3, 3), trainable=False, activation='relu', padding='same', name='block3_conv2')(x3)
    x3 = Conv2D(256, (3, 3), trainable=False, activation='relu', padding='same', name='block3_conv3')(x3)
    x3_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x3)
    # 256, 28, 28

    # Block 4
    x4 = Conv2D(512, (3, 3), trainable=False, activation='relu', padding='same', name='block4_conv1')(x3_pool)
    x4 = Conv2D(512, (3, 3), trainable=False, activation='relu', padding='same', name='block4_conv2')(x4)
    x4 = Conv2D(512, (3, 3), trainable=False, activation='relu', padding='same', name='block4_conv3')(x4)
    x4_pool = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x4)
    # 512, 14, 14

    # Block 5
    x5 = Conv2D(512, (3, 3), trainable=False, activation='relu', padding='same', name='block5_conv1')(x4_pool)
    x5 = Conv2D(512, (3, 3), trainable=False, activation='relu', padding='same', name='block5_conv2')(x5)
    x5 = Conv2D(512, (3, 3), trainable=False, activation='relu', padding='same', name='block5_conv3')(x5)

    # Upsampling
    x6 = UpSampling2D((2,2))(x5)
    x6 = Concatenate(axis=-1)([x6, x4])
    x6 = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer)(x6)
    x6 = BatchNormalization()(x6)
    x6 = activation()(x6)
    x6 = Conv2D(512, (3, 3), padding='same', kernel_regularizer=regularizer)(x6)
    x6 = BatchNormalization()(x6)
    x6 = activation()(x6)

    x7 = UpSampling2D((2,2))(x6)
    x7 = Concatenate(axis=-1)([x7, x3])
    x7 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer)(x7)
    x7 = BatchNormalization()(x7)
    x7 = activation()(x7)
    x7 = Conv2D(256, (3, 3), padding='same', kernel_regularizer=regularizer)(x7)
    x7 = BatchNormalization()(x7)
    x7 = activation()(x7)

    x8 = UpSampling2D((2,2))(x7)
    x8 = Concatenate(axis=-1)([x8, x2])
    x8 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer)(x8)
    x8 = BatchNormalization()(x8)
    x8 = activation()(x8)
    x8 = Conv2D(128, (3, 3), padding='same', kernel_regularizer=regularizer)(x8)
    x8 = BatchNormalization()(x8)
    x8 = activation()(x8)

    x9 = UpSampling2D((2,2))(x8)
    x9 = Concatenate(axis=-1)([x9, x1])
    x9 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer, name='s')(x9)
    x9 = BatchNormalization()(x9)
    x9 = activation()(x9)
    x9 = Conv2D(64, (3, 3), padding='same', kernel_regularizer=regularizer, name='conv2d_8')(x9)
    x9 = BatchNormalization()(x9)
    x9 = activation()(x9)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='conv2d_9')(x9)

    model = Model(img_input, classify, name='vgg16')
    model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coeff])

    return model
