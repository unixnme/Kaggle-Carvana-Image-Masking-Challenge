import keras
import keras.backend as K
from keras.layers import Lambda, Activation
from keras.layers.advanced_activations import PReLU

def leaky(alpha=0.1, beta=1.0):
    def activation():
        return Lambda(lambda x: K.minimum(0., x)*alpha + K.maximum(0., x)*beta)
    return activation

def relu(beta=1.0):
    def activation():
        return Lambda(lambda x: K.maximum(0., x)*beta)
    return activation

def elu():
    return Activation('elu')

def prelu():
    return PReLU(alpha_initializer='zeros', alpha_regularizer=None, alpha_constraint=None, shared_axes=None)

