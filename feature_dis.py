from keras.models import Model
from keras.layers import Input
from keras.layers.convolutional import Conv2D
from keras.applications import VGG19  ## keras pretraining model : for extract activated feature

def build_auxiliary_model(hr_shape):
    '''---------------<Inner structure modification : >-----------------'''
    oriVGG = VGG19(weights="imagenet", include_top=False)
    VGG_out = Conv2D(512, (3, 3),
                  padding='same',
                  name='block5_conv4')(oriVGG.get_layer('block5_conv3').output)
    ## Note : the block5_conv4 do not load the imagenet weight!!
    VGGBef= Model(inputs=oriVGG.input, outputs=VGG_out)
    '''-----------------------------------------------------------------'''
    
    VGGBef.outputs = [VGGBef.layers[-1].output]
    img = Input(shape=hr_shape)
    img_features = VGGBef(img)
    
    return Model(img, img_features)
