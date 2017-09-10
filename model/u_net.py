import keras
from keras.models import Model
from keras.layers import Input, concatenate, Conv2D, MaxPooling2D, Activation, UpSampling2D, BatchNormalization
from keras.optimizers import RMSprop

from model.losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff

def leaky():
    return keras.layers.advanced_activations.LeakyReLU(alpha=0.1)

def relu():
    return Activation('relu')

def get_unet_128(optimizer, input_shape=(128, 128, 3),
                 num_classes=1, regularizer=None, activation=leaky):
    inputs = Input(shape=input_shape)
    # 128

    down1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(inputs)
    down1 = BatchNormalization()(down1)
    down1 = activation()(down1)
    down1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = activation()(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = activation()(down2)
    down2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = activation()(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = activation()(down3)
    down3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = activation()(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = activation()(down4)
    down4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = activation()(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), kernel_regularizer=regularizer, padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = activation()(center)
    center = Conv2D(1024, (3, 3), kernel_regularizer=regularizer, padding='same')(center)
    center = BatchNormalization()(center)
    center = activation()(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    # 128

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up1)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coeff])

    return model


def get_unet_256(optimizer, input_shape=(256, 256, 3),
                 num_classes=1, regularizer=None, activation=relu):
    inputs = Input(shape=input_shape)
    # 256

    down0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(inputs)
    down0 = BatchNormalization()(down0)
    down0 = activation()(down0)
    down0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = activation()(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = activation()(down1)
    down1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = activation()(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = activation()(down2)
    down2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = activation()(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = activation()(down3)
    down3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = activation()(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = activation()(down4)
    down4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = activation()(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), kernel_regularizer=regularizer, padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = activation()(center)
    center = Conv2D(1024, (3, 3), kernel_regularizer=regularizer, padding='same')(center)
    center = BatchNormalization()(center)
    center = activation()(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = activation()(up0)
    up0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = activation()(up0)
    up0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = activation()(up0)
    # 256

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coeff])

    return model


def get_unet_512(optimizer, input_shape=(512, 512, 3),
                 num_classes=1, regularizer=None, activation=relu):
    inputs = Input(shape=input_shape)
    # 512

    down0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(inputs)
    down0a = BatchNormalization()(down0a)
    down0a = activation()(down0a)
    down0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = activation()(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = activation()(down0)
    down0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = activation()(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = activation()(down1)
    down1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = activation()(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = activation()(down2)
    down2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = activation()(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = activation()(down3)
    down3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = activation()(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = activation()(down4)
    down4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = activation()(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), kernel_regularizer=regularizer, padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = activation()(center)
    center = Conv2D(1024, (3, 3), kernel_regularizer=regularizer, padding='same')(center)
    center = BatchNormalization()(center)
    center = activation()(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = activation()(up0)
    up0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = activation()(up0)
    up0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = activation()(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = activation()(up0a)
    up0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = activation()(up0a)
    up0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = activation()(up0a)
    # 512

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0a)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coeff])

    return model


def get_unet_1024(optimizer, input_shape=(1024, 1024, 3),
                  num_classes=1, regularizer=None, activation=leaky):
    inputs = Input(shape=input_shape)
    # 1024

    down0b = Conv2D(8, (3, 3), kernel_regularizer=regularizer, padding='same')(inputs)
    down0b = BatchNormalization()(down0b)
    down0b = activation()(down0b)
    down0b = Conv2D(8, (3, 3), kernel_regularizer=regularizer, padding='same')(down0b)
    down0b = BatchNormalization()(down0b)
    down0b = activation()(down0b)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # 512

    down0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(down0b_pool)
    down0a = BatchNormalization()(down0a)
    down0a = activation()(down0a)
    down0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = activation()(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = activation()(down0)
    down0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = activation()(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = activation()(down1)
    down1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = activation()(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = activation()(down2)
    down2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = activation()(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = activation()(down3)
    down3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = activation()(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = activation()(down4)
    down4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = activation()(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), kernel_regularizer=regularizer, padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = activation()(center)
    center = Conv2D(1024, (3, 3), kernel_regularizer=regularizer, padding='same')(center)
    center = BatchNormalization()(center)
    center = activation()(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = activation()(up0)
    up0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = activation()(up0)
    up0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = activation()(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = activation()(up0a)
    up0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = activation()(up0a)
    up0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = activation()(up0a)
    # 512

    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(8, (3, 3), kernel_regularizer=regularizer, padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = activation()(up0b)
    up0b = Conv2D(8, (3, 3), kernel_regularizer=regularizer, padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = activation()(up0b)
    up0b = Conv2D(8, (3, 3), kernel_regularizer=regularizer, padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = activation()(up0b)
    # 1024

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0b)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coeff])

    return model



def get_unet_2048(optimizer, input_shape=(2048, 2048, 3),
                  num_classes=1, regularizer=None, activation=leaky):
    inputs = Input(shape=input_shape)
    # 2048

    down0c = Conv2D(4, (3, 3), kernel_regularizer=regularizer, padding='same')(inputs)
    down0c = BatchNormalization()(down0c)
    down0c = activation()(down0c)
    down0c = Conv2D(4, (3, 3), kernel_regularizer=regularizer, padding='same')(down0c)
    down0c = BatchNormalization()(down0c)
    down0c = activation()(down0c)
    down0c_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0c)
    #1024

    down0b = Conv2D(8, (3, 3), kernel_regularizer=regularizer, padding='same')(down0c_pool)
    down0b = BatchNormalization()(down0b)
    down0b = activation()(down0b)
    down0b = Conv2D(8, (3, 3), kernel_regularizer=regularizer, padding='same')(down0b)
    down0b = BatchNormalization()(down0b)
    down0b = activation()(down0b)
    down0b_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0b)
    # 512

    down0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(down0b_pool)
    down0a = BatchNormalization()(down0a)
    down0a = activation()(down0a)
    down0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(down0a)
    down0a = BatchNormalization()(down0a)
    down0a = activation()(down0a)
    down0a_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0a)
    # 256

    down0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(down0a_pool)
    down0 = BatchNormalization()(down0)
    down0 = activation()(down0)
    down0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(down0)
    down0 = BatchNormalization()(down0)
    down0 = activation()(down0)
    down0_pool = MaxPooling2D((2, 2), strides=(2, 2))(down0)
    # 128

    down1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(down0_pool)
    down1 = BatchNormalization()(down1)
    down1 = activation()(down1)
    down1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(down1)
    down1 = BatchNormalization()(down1)
    down1 = activation()(down1)
    down1_pool = MaxPooling2D((2, 2), strides=(2, 2))(down1)
    # 64

    down2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(down1_pool)
    down2 = BatchNormalization()(down2)
    down2 = activation()(down2)
    down2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(down2)
    down2 = BatchNormalization()(down2)
    down2 = activation()(down2)
    down2_pool = MaxPooling2D((2, 2), strides=(2, 2))(down2)
    # 32

    down3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(down2_pool)
    down3 = BatchNormalization()(down3)
    down3 = activation()(down3)
    down3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(down3)
    down3 = BatchNormalization()(down3)
    down3 = activation()(down3)
    down3_pool = MaxPooling2D((2, 2), strides=(2, 2))(down3)
    # 16

    down4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(down3_pool)
    down4 = BatchNormalization()(down4)
    down4 = activation()(down4)
    down4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(down4)
    down4 = BatchNormalization()(down4)
    down4 = activation()(down4)
    down4_pool = MaxPooling2D((2, 2), strides=(2, 2))(down4)
    # 8

    center = Conv2D(1024, (3, 3), kernel_regularizer=regularizer, padding='same')(down4_pool)
    center = BatchNormalization()(center)
    center = activation()(center)
    center = Conv2D(1024, (3, 3), kernel_regularizer=regularizer, padding='same')(center)
    center = BatchNormalization()(center)
    center = activation()(center)
    # center

    up4 = UpSampling2D((2, 2))(center)
    up4 = concatenate([down4, up4], axis=3)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    up4 = Conv2D(512, (3, 3), kernel_regularizer=regularizer, padding='same')(up4)
    up4 = BatchNormalization()(up4)
    up4 = activation()(up4)
    # 16

    up3 = UpSampling2D((2, 2))(up4)
    up3 = concatenate([down3, up3], axis=3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    up3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(up3)
    up3 = BatchNormalization()(up3)
    up3 = activation()(up3)
    # 32

    up2 = UpSampling2D((2, 2))(up3)
    up2 = concatenate([down2, up2], axis=3)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    up2 = Conv2D(128, (3, 3), kernel_regularizer=regularizer, padding='same')(up2)
    up2 = BatchNormalization()(up2)
    up2 = activation()(up2)
    # 64

    up1 = UpSampling2D((2, 2))(up2)
    up1 = concatenate([down1, up1], axis=3)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    up1 = Conv2D(64, (3, 3), kernel_regularizer=regularizer, padding='same')(up1)
    up1 = BatchNormalization()(up1)
    up1 = activation()(up1)
    # 128

    up0 = UpSampling2D((2, 2))(up1)
    up0 = concatenate([down0, up0], axis=3)
    up0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = activation()(up0)
    up0 = Conv2D(32, (3, 3), kernel_regularizer=regularizer, padding='same')(up0)
    up0 = BatchNormalization()(up0)
    up0 = activation()(up0)
    # 256

    up0a = UpSampling2D((2, 2))(up0)
    up0a = concatenate([down0a, up0a], axis=3)
    up0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = activation()(up0a)
    up0a = Conv2D(16, (3, 3), kernel_regularizer=regularizer, padding='same')(up0a)
    up0a = BatchNormalization()(up0a)
    up0a = activation()(up0a)
    # 512

    up0b = UpSampling2D((2, 2))(up0a)
    up0b = concatenate([down0b, up0b], axis=3)
    up0b = Conv2D(8, (3, 3), kernel_regularizer=regularizer, padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = activation()(up0b)
    up0b = Conv2D(8, (3, 3), kernel_regularizer=regularizer, padding='same')(up0b)
    up0b = BatchNormalization()(up0b)
    up0b = activation()(up0b)
    # 1024

    up0c = UpSampling2D((2, 2))(up0b)
    up0c = concatenate([down0c, up0c], axis=3)
    up0c = Conv2D(4, (3, 3), kernel_regularizer=regularizer, padding='same')(up0c)
    up0c = BatchNormalization()(up0c)
    up0c = activation()(up0c)
    up0c = Conv2D(4, (3, 3), kernel_regularizer=regularizer, padding='same')(up0c)
    up0c = BatchNormalization()(up0c)
    up0c = activation()(up0c)
    # 2048

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid')(up0c)

    model = Model(inputs=inputs, outputs=classify)

    model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coeff])

    return model
