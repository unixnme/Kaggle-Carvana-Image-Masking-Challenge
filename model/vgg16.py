import keras
from keras.models import Model
from keras.layers import Conv2D, Input, MaxPooling2D, UpSampling2D, concatenate
from keras.optimizers import SGD
from u_net import relu
from model.losses import bce_dice_loss, dice_coeff

def get_vgg16(input_shape=(224,224,3), regularizer=None, activation=relu, num_classes=1, optimizer=SGD()):
    img_input = Input(shape=input_shape)

    # Block 1 --> x1
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv1', trainable=False)(img_input)
    x1 = Conv2D(64, (3, 3), activation='relu', padding='same', name='block1_conv2', trainable=False)(x1)
    x1p = MaxPooling2D((2, 2), strides=(2, 2), name='block1_pool')(x1)

    # Block 2 --> x2
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv1', trainable=False)(x1p)
    x2 = Conv2D(128, (3, 3), activation='relu', padding='same', name='block2_conv2', trainable=False)(x2)
    x2p = MaxPooling2D((2, 2), strides=(2, 2), name='block2_pool')(x2)

    # Block 3 --> x4
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv1', trainable=False)(x2p)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv2', trainable=False)(x3)
    x3 = Conv2D(256, (3, 3), activation='relu', padding='same', name='block3_conv3', trainable=False)(x3)
    x3p = MaxPooling2D((2, 2), strides=(2, 2), name='block3_pool')(x3)

    # Block 4 --> x8
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv1', trainable=False)(x3p)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv2', trainable=False)(x4)
    x4 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block4_conv3', trainable=False)(x4)
    x4p = MaxPooling2D((2, 2), strides=(2, 2), name='block4_pool')(x4)

    # Block 5 --> x16
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv1', trainable=False)(x4p)
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv2', trainable=False)(x5)
    x5 = Conv2D(512, (3, 3), activation='relu', padding='same', name='block5_conv3', trainable=False)(x5)
    x5u = UpSampling2D()(x5)

    # Block 6 --> x8
    x6c = concatenate([x5u, x3p])
    x6 = Conv2D(512, (1, 1), activation=activation(), padding='same', name='block6_conv1')(x6c)
    x6 = Conv2D(512, (3, 3), activation=activation(), padding='same', name='block6_conv2')(x6)
    x6 = Conv2D(512, (3, 3), activation=activation(), padding='same', name='block6_conv3')(x6)
    x6 = Conv2D(512, (3, 3), activation=activation(), padding='same', name='block6_conv4')(x6)
    x6u = UpSampling2D()(x6)

    # Block 7 --> x4
    x7c = concatenate([x6u, x2p])
    x7 = Conv2D(256, (1, 1), activation=activation(), padding='same', name='block7_conv1')(x7c)
    x7 = Conv2D(256, (3, 3), activation=activation(), padding='same', name='block7_conv2')(x7)
    x7 = Conv2D(256, (3, 3), activation=activation(), padding='same', name='block7_conv3')(x7)
    x7 = Conv2D(256, (3, 3), activation=activation(), padding='same', name='block7_conv4')(x7)
    x7u = UpSampling2D()(x7)

    # Block 8 --> x2
    x8c = concatenate([x7u, x1p])
    x8 = Conv2D(128, (1, 1), activation=activation(), padding='same', name='block8_conv1')(x8c)
    x8 = Conv2D(128, (3, 3), activation=activation(), padding='same', name='block8_conv2')(x8)
    x8 = Conv2D(128, (3, 3), activation=activation(), padding='same', name='block8_conv3')(x8)
    x8u = UpSampling2D()(x8)

    # Block 9 --> x1
    x9c = concatenate([x8u, img_input])
    x9 = Conv2D(64, (1, 1), activation=activation(), padding='same', name='block9_conv1')(x9c)
    x9 = Conv2D(64, (3, 3), activation=activation(), padding='same', name='block9_conv2')(x9)
    x9 = Conv2D(64, (3, 3), activation=activation(), padding='same', name='block9_conv3')(x9)

    classify = Conv2D(num_classes, (1, 1), activation='sigmoid', name='classify')(x9)
    model = Model(inputs=img_input, outputs=classify)

    model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coeff])

    return model