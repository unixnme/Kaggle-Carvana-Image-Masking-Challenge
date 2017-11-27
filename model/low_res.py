from keras.models import Model
from keras.layers import Conv2D, concatenate, BatchNormalization, Input, MaxPooling2D, AveragePooling2D, UpSampling2D, Add
from keras.layers.merge import Concatenate
from model.u_net import relu, leaky

def block(x, name, kernel=3, num_conv = 3, filter=4, dilation=1, regularizer=None, activation=relu, BN=False, initializer='glorot_uniform',
        use_bias=True):
    for idx in range(num_conv):
        x = Conv2D(filter, kernel, padding='same', dilation_rate=dilation, kernel_regularizer=regularizer, bias_regularizer=regularizer, name=name+'_conv'+str(idx+1), kernel_initializer=initializer)(x)
        x = activation()(x)
        if BN is True:
            x = BatchNormalization(name=name+'_BN'+str(idx+1))(x)
    return x

def create_model(shape, num_blocks=3, num_conv = 3, kernel=3, filter=[4,8,16,8,4], encoding_dilation=1, decoding_dilation=1, regularizer=None, activation=relu, BN=False, pooling='max', initializer='glorot_uniform', use_bias=True):
    img_in = Input(shape=shape)
    x = img_in
    x = BatchNormalization()(x)
    index = 0

    # encoding
    encoders = []
    for i in range(num_blocks):
        x = block(x, name='down_block'+str(i+1), kernel=kernel, num_conv=num_conv, filter=filter[i], dilation=encoding_dilation, regularizer=regularizer, activation=activation, BN=BN, initializer=initializer, use_bias=use_bias)
        encoders.append(x)
        if pooling == 'max':
            x = MaxPooling2D(padding='same')(x)
        elif pooling == 'average':
            x = AveragePooling2D(padding='same')(x)
        else:
            raise Exception('pooling must be "max" or "average"')
        index += 1

    # processing
    x = block(x, name='center_block', kernel=kernel, num_conv=num_conv, filter=filter[index], dilation=encoding_dilation, regularizer=regularizer, activation=activation, BN=BN, initializer=initializer, use_bias=use_bias)
    index += 1

    # decoding
    for i in range(num_blocks):
        x = UpSampling2D()(x)
        x = block(x, name='up_block'+str(num_blocks-i), kernel=kernel, num_conv=num_conv,  filter=filter[index], dilation=decoding_dilation, regularizer=regularizer, activation=activation, BN=BN, initializer=initializer, use_bias=use_bias)
        x = Concatenate(axis=-1)([x, encoders.pop()])
        x = Conv2D(filter[index], 1,
                   name='up_block'+str(num_blocks-i)+'_concate_conv',
                   kernel_regularizer=regularizer, bias_regularizer=regularizer,
                   kernel_initializer=initializer, use_bias=use_bias,
                   padding='same')(x)
        x = activation()(x)
        if BN is True:
            x = BatchNormalization(name='up_block'+str(num_blocks-i)+'_concate_BN')(x)
        index += 1

    classify = Conv2D(1, 1, name='classify', activation='sigmoid')(x)
    return Model(inputs=img_in, outputs=classify)

if __name__ == '__main__':
    model = create_model((256, 256, 3), num_blocks=3, kernel=3, filter=4, dilation=1, regularizer=None, activation=relu, BN=True, pooling='max')
    model.summary()
    from keras.utils.vis_utils import plot_model
    plot_model(model, show_shapes=True)
