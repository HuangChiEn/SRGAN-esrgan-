from keras.models import Model
from keras.layers import Input

def package_dis_gen(auxMod, discriminator, ):
    auxMod.trainable = False
    discriminator.trainable = False
    Input()
    mod = Model([img_lr, img_hr], [validity, fake_features])