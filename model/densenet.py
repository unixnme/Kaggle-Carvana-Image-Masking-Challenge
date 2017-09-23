import keras
from keras.layers import Conv2D, Input, Activation, UpSampling2D, AveragePooling2D, concatenate
from u_net import relu, leaky
from losses import bce_dice_loss, dice_loss, weighted_bce_dice_loss, weighted_dice_loss, dice_coeff, sparse_cce_dice_loss
from keras.optimizers import SGD

def dense_net_128(input_shape=(256,256,3), optimizer=SGD(), activation=relu, regularizer=None, num_classes=2):
    inputs = Input(shape=input_shape)

    x1 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(inputs)
    x1 = activation()(x1)
    x1 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x1)
    x1 = activation()(x1)
    x1_pool = AveragePooling2D((2, 2))(x1)

    input2 = AveragePooling2D((2,2))(inputs)
    x2c = concatenate([x1_pool, input2], axis=-1)
    x2 = Conv2D(256, (1, 1), kernel_regularizer=regularizer, padding='same')(x2c)
    x2 = activation()(x2)
    x2 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x2)
    x2 = activation()(x2)
    x2 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x2)
    x2 = activation()(x2)
    x2_pool = AveragePooling2D((2, 2))(x2)

    input4 = AveragePooling2D((4,4))(inputs)
    x1_pool4 = AveragePooling2D((4, 4))(x1)
    x3c = concatenate([x2_pool, input4, x1_pool4], axis=-1)
    x3 = Conv2D(256, (1, 1), kernel_regularizer=regularizer, padding='same')(x3c)
    x3 = activation()(x3)
    x3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x3)
    x3 = activation()(x3)
    x3 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x3)
    x3 = activation()(x3)
    x3_pool = AveragePooling2D((2, 2))(x3)

    input8 = AveragePooling2D((8,8))(inputs)
    x1_pool8 = AveragePooling2D((8, 8))(x1)
    x2_pool4 = AveragePooling2D((4, 4))(x2)
    x4c = concatenate([x3_pool, input8, x1_pool8, x2_pool4], axis=-1)
    x4 = Conv2D(256, (1, 1), kernel_regularizer=regularizer, padding='same')(x4c)
    x4 = activation()(x4)
    x4 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x4)
    x4 = activation()(x4)
    x4 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x4)
    x4 = activation()(x4)
    x4_pool = AveragePooling2D((2, 2))(x4)

    input16 = AveragePooling2D((16,16))(inputs)
    x1_pool16 = AveragePooling2D((16, 16))(x1)
    x2_pool8 = AveragePooling2D((8, 8))(x2)
    x3_pool4 = AveragePooling2D((4,4))(x3)
    x5c = concatenate([x4_pool, input16, x1_pool16, x2_pool8, x3_pool4], axis=-1)
    x5 = Conv2D(256, (1, 1), kernel_regularizer=regularizer, padding='same')(x5c)
    x5 = activation()(x5)
    x5 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x5)
    x5 = activation()(x5)
    x5 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x5)
    x5 = activation()(x5)
    x5_pool = AveragePooling2D((2, 2))(x5)

    input32 = AveragePooling2D((32,32))(inputs)
    x1_pool32 = AveragePooling2D((32, 32))(x1)
    x2_pool16 = AveragePooling2D((16, 16))(x2)
    x3_pool8 = AveragePooling2D((8,8))(x3)
    x4_pool4 = AveragePooling2D((4,4))(x4)
    x6c = concatenate([x5_pool, input32, x1_pool32, x2_pool16, x3_pool8, x4_pool4], axis=-1)
    x6 = Conv2D(256, (1, 1), kernel_regularizer=regularizer, padding='same')(x6c)
    x6 = activation()(x6)
    x6 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x6)
    x6 = activation()(x6)
    x6 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x6)
    x6 = activation()(x6)
    x6 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x6)
    x6 = activation()(x6)

    x7 = UpSampling2D()(x6)
    x7 = concatenate([x7, x5c])
    x7 = Conv2D(256, (1,1), kernel_regularizer=regularizer, padding='same')(x7)
    x7 = activation()(x7)
    x7 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x7)
    x7 = activation()(x7)
    x7 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x7)
    x7 = activation()(x7)

    x8 = UpSampling2D()(x7)
    x8 = concatenate([x8, x4c])
    x8 = Conv2D(256, (1,1), kernel_regularizer=regularizer, padding='same')(x8)
    x8 = activation()(x8)
    x8 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x8)
    x8 = activation()(x8)
    x8 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x8)
    x8 = activation()(x8)

    x9 = UpSampling2D()(x8)
    x9 = concatenate([x9, x3c])
    x9 = Conv2D(256, (1,1), kernel_regularizer=regularizer, padding='same')(x9)
    x9 = activation()(x9)
    x9 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x9)
    x9 = activation()(x9)
    x9 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x9)
    x9 = activation()(x9)

    x10 = UpSampling2D()(x9)
    x10 = concatenate([x10, x2c])
    x10 = Conv2D(256, (1,1), kernel_regularizer=regularizer, padding='same')(x10)
    x10 = activation()(x10)
    x10 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x10)
    x10 = activation()(x10)
    x10 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x10)
    x10 = activation()(x10)

    x11 = UpSampling2D()(x10)
    x11 = concatenate([x11, x1, inputs])
    x11 = Conv2D(256, (1,1), kernel_regularizer=regularizer, padding='same')(x11)
    x11 = activation()(x11)
    x11 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x11)
    x11 = activation()(x11)
    x11 = Conv2D(256, (3, 3), kernel_regularizer=regularizer, padding='same')(x11)
    x11 = activation()(x11)
    x11 = Conv2D(1, (3, 3), kernel_regularizer=regularizer, padding='same')(x11)

    classify = Activation('sigmoid')(x11)

    model = keras.models.Model(inputs=inputs, outputs=classify)
    model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coeff])

    return model




def densenet(input_shape=(256,256,3), optimizer=SGD(), activation=relu, regularizer=None, num_classes=1):
    def dense_layer(x, xc):
        zc = concatenate([x, xc])
        z = Conv2D(32, [1,1], padding='same', kernel_regularizer=regularizer)(zc)
        z = activation()(z)
        z = Conv2D(32, [3, 3], padding='same', kernel_regularizer=regularizer)(z)
        z = activation()(z)
        z = Conv2D(32, [3, 3], padding='same', kernel_regularizer=regularizer)(z)
        z = activation()(z)
        z = Conv2D(32, [3, 3], padding='same', kernel_regularizer=regularizer)(z)
        z = activation()(z)
        return z, zc

    img_input = Input(shape=input_shape)
    x0 = Conv2D(32, [3,3], padding='same', kernel_regularizer=regularizer)(img_input)
    x0 = activation()(x0)
    x0 = Conv2D(32, [3,3], padding='same', kernel_regularizer=regularizer)(x0)
    x0 = activation()(x0)
    x0 = Conv2D(32, [3,3], padding='same', kernel_regularizer=regularizer)(x0)
    x0 = activation()(x0)

    x1 = Conv2D(32, [3,3], padding='same', kernel_regularizer=regularizer)(x0)
    x1 = activation()(x1)
    x1 = Conv2D(32, [3, 3], padding='same', kernel_regularizer=regularizer)(x1)
    x1 = activation()(x1)
    x1 = Conv2D(32, [3, 3], padding='same', kernel_regularizer=regularizer)(x1)
    x1 = activation()(x1)

    x = x1
    xc = x0

    for i in range(41):
        x, xc = dense_layer(x, xc)

    x = Conv2D(num_classes, [1,1], padding='same', kernel_regularizer=regularizer)(x)
    x = Activation('sigmoid')(x)

    model = keras.models.Model(inputs=img_input, outputs=x)
    model.compile(optimizer=optimizer, loss=bce_dice_loss, metrics=[dice_coeff])

    return model