from keras.models import Model
from keras.layers import Lambda, Add, Input
from keras.layers.advanced_activations import PReLU, LeakyReLU
from keras.layers.convolutional import UpSampling2D, Conv2D, Conv2DTranspose

'''keras package : the high-level module for deep learning 
==============================<Eliminated subpixel convolution block>===============================

    def SubpixelConv2D(self, input_shape, scale=2):
        def subpixel_shape(input_shape):
            dims = [input_shape[0], 
                    input_shape[1] * scale,
                    input_shape[2] * scale,
                    int(input_shape[3]/(scale ** 2))]
            output_shape = tuple(dims)
            return output_shape
        
        return Lambda(lambda x: tf.depth_to_space(x, scale), output_shape=subpixel_shape)  ## name='subpixel'
    
    @ For activate the subpixel function,
            please insert the following code into the upsample function.
    ##x = self.SubpixelConv2D(x.shape)(x);x = Activation('tanh')(x)
    
    ===================================================================================================='''

## parameter order may adjustment
def build_generator(lr_shape, num_of_filts, num_of_RRDB, num_of_DB, resScal, upScalar, is_train=True):
    def dense_block(input_tensor, filters, scale=0.2):
        skip_connt = x_1 = input_tensor
    
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(input_tensor)
        x = LeakyReLU(alpha=0.2)(x)
        x = x_2 = Add()([x_1, x])
    
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = x_3 = Add()([x_1, x_2, x])
    
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = x_4 = Add()([x_1, x_2, x_3, x])
    
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        x = LeakyReLU(alpha=0.2)(x)
        x = Add()([x_1, x_2, x_3, x_4, x])
    
        x = Conv2D(filters=filters, kernel_size=3, strides=1, padding='same')(x)
        ## Residual Scaling && skip connection in dense block : 
        x = Lambda(lambda x: x * resScal)(x)
        x = Add()([skip_connt, x])
        return x
    ## may use ConvTranspose2D() upsampling ..
    def upsample(input_tensor, filters):  
        x = Conv2D(filters=filters*4, kernel_size=3, strides=1, padding='same')(input_tensor)
        ##x = self.SubpixelConv2D(x.shape)(x)
        ##x = Activation('relu')(x)
        x = UpSampling2D(size=2, interpolation='bilinear')(x)
        ##x = UpSampling2D(size=2)(x)
        ##x = Conv2DTranspose(1, (2, 2), strides=(2, 2), padding='same')(x)
        x = PReLU(shared_axes=[1, 2])(x)     
        return x
    
    
    inputs = Input(shape=lr_shape)
    ## < Feature extractor structure/ > 
    x = Conv2D(filters=num_of_filts, kernel_size=3, strides=1, padding='same')(inputs)
    out_skip = in_skip = x = LeakyReLU(alpha=0.2)(x)
    
    for _ in range(num_of_RRDB):
        ## Residual in Residual Dense Block : 
        for _ in range(num_of_DB):
            ## Residual Dense Blocks :
            x = dense_block(input_tensor=x, filters=num_of_filts)      
        ## out block process : (scaling and add)
        x = Lambda(lambda x: x * resScal)(x)
        x = in_skip = Add()([in_skip, x])
        
    x = Conv2D(filters=num_of_filts, kernel_size=3, strides=1, padding='same')(x)
    ## < /Feature extractor structure > 
    ## Add Layer 
    '''Source code of edsr do not contain residual scaling : '''
    x = Lambda(lambda x: x * resScal)(x)  ## residual scaling beta=0.8
    x = Add()([out_skip, x])
    
    for _ in range(upScalar):
        x = upsample(x, num_of_filts)

    x = Conv2D(filters=num_of_filts, kernel_size=3, strides=1, padding='same')(x)
    x = LeakyReLU(alpha=0.2)(x)
    x = Conv2D(filters=3, kernel_size=3, strides=1, padding='same')(x)
    
    return Model(inputs=inputs, outputs=x)
    