'''keras package : the high-level module for deep learning '''
from keras.models import Model
from keras.layers import Input, Dense, BatchNormalization, Add
from keras.layers.advanced_activations import LeakyReLU
from keras.layers.convolutional import Conv2D

def build_discriminator(hr_shape, num_of_filts):

    def d_block(layer_input, filters, strides=1, bn=True, kerSiz=3):
        """Discriminator layer"""
        d = Conv2D(filters, kernel_size=kerSiz, strides=strides, padding='same')(layer_input)
        d = LeakyReLU(alpha=0.2)(d)
        if bn:
            d = BatchNormalization(momentum=0.8)(d)
        return d
    
    d0 = Input(shape=hr_shape)

    d1 = d_block(d0, num_of_filts, bn=False)
    d2 = d_block(d1, num_of_filts, strides=2, kerSiz=4)
    d3 = d_block(d2, num_of_filts*2)
    d4 = d_block(d3, num_of_filts*2, strides=2, kerSiz=4)
    d5 = d_block(d4, num_of_filts*4)
    d6 = d_block(d5, num_of_filts*4, strides=2, kerSiz=4)
    d7 = d_block(d6, num_of_filts*8)
    d8 = d_block(d7, num_of_filts*8, strides=2, kerSiz=4)
    
    d9 = Dense(num_of_filts*16)(d8)
    d10 = LeakyReLU(alpha=0.2)(d9)
    validity = Dense(1, activation='sigmoid')(d10)
    return Model(d0, validity)