import numpy as np 
np.random.seed(1337)
import theano.tensor as T
from keras.layers.advanced_activations import ELU
from keras.layers.convolutional import Convolution2D, MaxPooling2D, UpSampling2D
from keras.layers.normalization import BatchNormalization
from theano.compile.nanguardmode import NanGuardMode
THEANO_FLAGS=mode=NanGuardMode
from keras.layers.core import Dropout
from keras.optimizers import Adam
from keras.layers import merge, Input
from keras.models import Model

import config as cfg

#################################
# LOSS FUNCTION
#################################
delta = 0.1
def huber(target, output):
    d = target - output
    a = .5 * d**2
    b = delta * (abs(d) - delta / 2.)
    l = T.switch(abs(d) <= delta, a, b)
    return l.sum()


#################################
# MODELS
#################################
def Unet(shp=cfg.shape_train_data, weights_path=cfg.path_load_weights):
    print('-- Current file: model.py -- Unet model')
    
    act='relu'    
    
    inputs = Input((shp))
    conv1 = Convolution2D(32, 3, 3, activation=act, border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation=act, border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation=act, border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation=act, border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation=act, border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation=act, border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation=act, border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation=act, border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation=act, border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation=act, border_mode='same')(conv5)

    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation=act, border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation=act, border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation=act, border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation=act, border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation=act, border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation=act, border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation=act, border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation=act, border_mode='same')(conv9)

    conv10 = Convolution2D(3, 1, 1, activation='sigmoid')(conv9)

    model = Model(input=inputs, output=conv10)
    
    if weights_path <> '':
        model.load_weights(weights_path)

    #model.compile(optimizer=Adam(lr=cfg.learning_rate), loss=huber) 
    model.compile(optimizer='sgd', loss='mse') 
    
    return model

 
def Unet_HC(shp=cfg.shape_train_data, weights_path=cfg.path_load_weights):
    print('-- Current file: model.py -- Unet_HC model')
    
    act='elu'    
    
    inputs = Input(shp, name='main_input')
    
    conv1 = Convolution2D(32, 3, 3, activation=act, border_mode='same')(inputs)
    conv1 = Convolution2D(32, 3, 3, activation=act, border_mode='same')(conv1)
    pool1 = MaxPooling2D(pool_size=(2, 2))(conv1)

    conv2 = Convolution2D(64, 3, 3, activation=act, border_mode='same')(pool1)
    conv2 = Convolution2D(64, 3, 3, activation=act, border_mode='same')(conv2)
    pool2 = MaxPooling2D(pool_size=(2, 2))(conv2)

    conv3 = Convolution2D(128, 3, 3, activation=act, border_mode='same')(pool2)
    conv3 = Convolution2D(128, 3, 3, activation=act, border_mode='same')(conv3)
    pool3 = MaxPooling2D(pool_size=(2, 2))(conv3)

    conv4 = Convolution2D(256, 3, 3, activation=act, border_mode='same')(pool3)
    conv4 = Convolution2D(256, 3, 3, activation=act, border_mode='same')(conv4)
    pool4 = MaxPooling2D(pool_size=(2, 2))(conv4)

    conv5 = Convolution2D(512, 3, 3, activation=act, border_mode='same')(pool4)
    conv5 = Convolution2D(512, 3, 3, activation=act, border_mode='same')(conv5)
    
    #Hypercolumns
    #hc_conv5 = UpSampling2D(size=(16, 16))(conv5) #8x8, f = 16
    #hc_conv4 = UpSampling2D(size=(8, 8))(conv4) #16x16, f = 8
    hc_conv3 = UpSampling2D(size=(4, 4))(conv3) #32x32, f = 4
    hc_conv2 = UpSampling2D(size=(2, 2))(conv2) #64x64, f = 2
    hc = merge([conv1, hc_conv2, hc_conv3], mode='concat', concat_axis=1) #(None, 992, 128, 128)
    hc_red_conv1 = Convolution2D(128, 3, 3, border_mode='same', init='he_normal', activation=act)(hc)
    hc_red_conv2 = Convolution2D(64, 3, 3, border_mode='same', init='he_normal', activation=act)(hc_red_conv1)
    hc_red_conv3 = Convolution2D(shp[0], 1, 1, init='he_normal', activation='linear', name='aux_output')(hc_red_conv2)
    
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = Convolution2D(256, 3, 3, activation=act, border_mode='same')(up6)
    conv6 = Convolution2D(256, 3, 3, activation=act, border_mode='same')(conv6)

    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = Convolution2D(128, 3, 3, activation=act, border_mode='same')(up7)
    conv7 = Convolution2D(128, 3, 3, activation=act, border_mode='same')(conv7)

    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = Convolution2D(64, 3, 3, activation=act, border_mode='same')(up8)
    conv8 = Convolution2D(64, 3, 3, activation=act, border_mode='same')(conv8)

    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = Convolution2D(32, 3, 3, activation=act, border_mode='same')(up9)
    conv9 = Convolution2D(32, 3, 3, activation=act, border_mode='same')(conv9)

    conv10 = Convolution2D(3, 1, 1, activation='linear', name='main_output')(conv9)

    model = Model(input=inputs, output=[conv10, hc_red_conv3])
    
    if weights_path <> '':
        model.load_weights(weights_path)
    
    model.compile(optimizer=Adam(lr=cfg.learning_rate), loss={'main_output': huber, 'aux_output': huber}, loss_weights={'main_output': 0.8, 'aux_output': 0.2})
     
    return model


def inception_block(inputs, depth, batch_mode=0, split=False):
    assert depth % 16 == 0
    
    # Branch 1
    b1_1 = Convolution2D(depth/4, 1, 1, init='he_normal', border_mode='same')(inputs)
    b2_1 = Convolution2D(depth/8*3, 1, 1, init='he_normal', border_mode='same')(inputs)
    
    # Branch 2
    b2_1 = ELU()(b2_1)
    if split:
        b2_2 = Convolution2D(depth/2, 1, 3, init='he_normal', border_mode='same')(b2_1)
        b2_2 = BatchNormalization(mode=batch_mode, axis=1)(b2_2)
        b2_2 = ELU()(b2_2)
        b2_3 = Convolution2D(depth/2, 3, 1, init='he_normal', border_mode='same')(b2_2)
    else:
        b2_3 = Convolution2D(depth/2, 3, 3, init='he_normal', border_mode='same')(b2_1)
        
    # Branch 3
    b3_1 = Convolution2D(depth/16, 1, 1, init='he_normal', border_mode='same')(inputs)
    b3_1 = ELU()(b3_1)
    if split:
        b3_2 = Convolution2D(depth/8, 1, 5, init='he_normal', border_mode='same')(b3_1)
        b3_2 = BatchNormalization(mode=batch_mode, axis=1)(b3_2)
        b3_2 = ELU()(b3_2)
        b3_3 = Convolution2D(depth/8, 5, 1, init='he_normal', border_mode='same')(b3_2)
    else:
        b3_3 = Convolution2D(depth/8, 5, 5, init='he_normal', border_mode='same')(b3_1)
    
    # Branch 4
    b4_1 = MaxPooling2D(pool_size=(3,3), strides=(1,1), border_mode='same')(inputs)
    b4_2 = Convolution2D(depth/8, 1, 1, init='he_normal', border_mode='same')(b4_1)
    
    # Filter output
    res = merge([b1_1, b2_3, b3_3, b4_2], mode='concat', concat_axis=1)
    res = BatchNormalization(mode=batch_mode, axis=1)(res)
    res = ELU()(res)
    
    return res
    

def custom_convolution2D(nb_filter, nb_row, nb_col, border_mode='same', subsample=(1, 1)):
    def f(_input):
        conv = Convolution2D(nb_filter=nb_filter, nb_row=nb_row, nb_col=nb_col, subsample=subsample,
                              border_mode=border_mode)(_input)
        norm = BatchNormalization(mode=2, axis=1)(conv)
        return ELU()(norm)
    
    return f

    
def SUnet(shp=cfg.shape_train_data, weights_path=cfg.path_load_weights):
    print('-- Current file: model.py -- Unet model with inception modules')
    
    split = False
    
    # Encoder
    inputs = Input(shp)
    conv1 = inception_block(inputs, 32, batch_mode=2, split=split)
    
    pool1 = custom_convolution2D(32, 3, 3, border_mode='same', subsample=(2,2))(conv1)
    pool1 = Dropout(0.5)(pool1)
    
    conv2 = inception_block(pool1, 64, batch_mode=2, split=split)
    pool2 = custom_convolution2D(64, 3, 3, border_mode='same', subsample=(2,2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    
    conv3 = inception_block(pool2, 128, batch_mode=2, split=split)
    pool3 = custom_convolution2D(128, 3, 3, border_mode='same', subsample=(2,2))(conv3)
    pool3 = Dropout(0.5)(pool3)
     
    conv4 = inception_block(pool3, 256, batch_mode=2, split=split)
    pool4 = custom_convolution2D(256, 3, 3, border_mode='same', subsample=(2,2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    
    conv5 = inception_block(pool4, 512, batch_mode=2, split=split)
    conv5 = Dropout(0.5)(conv5)
    
    # Decoder
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = inception_block(up6, 256, batch_mode=2, split=split)
    conv6 = Dropout(0.5)(conv6)
    
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = inception_block(up7, 128, batch_mode=2, split=split)
    conv7 = Dropout(0.5)(conv7)
    
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = inception_block(up8, 64, batch_mode=2, split=split)
    conv8 = Dropout(0.5)(conv8)
    
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = inception_block(up9, 32, batch_mode=2, split=split)
    conv9 = Dropout(0.5)(conv9)

    conv10 = Convolution2D(shp[0], 1, 1, init='he_normal', activation='sigmoid')(conv9)

    model = Model(input=inputs, output=[conv10])
    
    if weights_path <> '':
        model.load_weights(weights_path)
    
    model.compile(loss=huber, optimizer=Adam(lr=cfg.learning_rate))
    
    return model
    
    
def HC(shp=cfg.shape_train_data, weights_path=cfg.path_load_weights):
    print('-- Current file: model.py -- HC model')
    
    split = True
    
    # Encoder
    inputs = Input(shp)
    conv1 = inception_block(inputs, 32, batch_mode=2, split=split)
    
    pool1 = custom_convolution2D(32, 3, 3, border_mode='same', subsample=(2,2))(conv1)
    pool1 = Dropout(0.5)(pool1)
    
    conv2 = inception_block(pool1, 64, batch_mode=2, split=split)
    pool2 = custom_convolution2D(64, 3, 3, border_mode='same', subsample=(2,2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    
    conv3 = inception_block(pool2, 128, batch_mode=2, split=split)
    pool3 = custom_convolution2D(128, 3, 3, border_mode='same', subsample=(2,2))(conv3)
    pool3 = Dropout(0.5)(pool3)
     
    conv4 = inception_block(pool3, 256, batch_mode=2, split=split)
    pool4 = custom_convolution2D(256, 3, 3, border_mode='same', subsample=(2,2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    
    conv5 = inception_block(pool4, 512, batch_mode=2, split=split)
    conv5 = Dropout(0.5)(conv5)
    
    # Hypercolumns
    hc_conv5 = UpSampling2D(size=(16, 16))(conv5) #8x8, f = 16
    hc_conv4 = UpSampling2D(size=(8, 8))(conv4) #16x16, f = 8
    hc_conv3 = UpSampling2D(size=(4, 4))(conv3) #32x32, f = 4
    hc_conv2 = UpSampling2D(size=(2, 2))(conv2) #64x64, f = 2
    
    hc = merge([conv1, hc_conv2, hc_conv3, hc_conv4, hc_conv5], mode='concat', concat_axis=1)
    #hc = merge([conv1, hc_conv2, hc_conv3], mode='concat', concat_axis=1)

    i1 = inception_block(hc, 128, batch_mode=2, split=split)
    i1 = Dropout(0.5)(i1)
    i2 = inception_block(i1, 64, batch_mode=2, split=split)
    i2 = Dropout(0.5)(i2)
    hc_red_conv3 = Convolution2D(shp[0], 1, 1, init='he_normal', activation='linear', name='aux_output')(i2)

    model = Model(input=inputs, output=[hc_red_conv3])
    
    if weights_path <> '':
        model.load_weights(weights_path)
    
    model.compile(loss=huber, optimizer=Adam(clipnorm=1.))    
    
    return model


def EURNet(shp=cfg.shape_train_data, weights_path=cfg.path_load_weights):
    print('-- Current file: model.py -- EURNet model')
    
    split = True
    
    #Encoder
    inputs = Input(shp, name='main_input')
    conv1 = inception_block(inputs, 32, batch_mode=2, split=split)
    
    pool1 = custom_convolution2D(32, 3, 3, border_mode='same', subsample=(2,2))(conv1)
    pool1 = Dropout(0.5)(pool1)
    
    conv2 = inception_block(pool1, 64, batch_mode=2, split=split)
    pool2 = custom_convolution2D(64, 3, 3, border_mode='same', subsample=(2,2))(conv2)
    pool2 = Dropout(0.5)(pool2)
    
    conv3 = inception_block(pool2, 128, batch_mode=2, split=split)
    pool3 = custom_convolution2D(128, 3, 3, border_mode='same', subsample=(2,2))(conv3)
    pool3 = Dropout(0.5)(pool3)
     
    conv4 = inception_block(pool3, 256, batch_mode=2, split=split)
    pool4 = custom_convolution2D(256, 3, 3, border_mode='same', subsample=(2,2))(conv4)
    pool4 = Dropout(0.5)(pool4)
    
    conv5 = inception_block(pool4, 512, batch_mode=2, split=split)
    conv5 = Dropout(0.5)(conv5)
    
    #Hypercolumns
    hc_conv3 = UpSampling2D(size=(4, 4))(conv3)
    hc_conv2 = UpSampling2D(size=(2, 2))(conv2)
    hc = merge([conv1, hc_conv2, hc_conv3], mode='concat', concat_axis=1)
    
    i1 = inception_block(hc, 128, batch_mode=2, split=split)
    i1 = Dropout(0.5)(i1)
    i2 = inception_block(i1, 64, batch_mode=2, split=split)
    i2 = Dropout(0.5)(i2)
    hc_red_conv3 = Convolution2D(shp[0], 1, 1, init='he_normal', activation='linear', name='aux_output')(i2) #W_regularizer=l2(0.01)
    
    #Decoder
    up6 = merge([UpSampling2D(size=(2, 2))(conv5), conv4], mode='concat', concat_axis=1)
    conv6 = inception_block(up6, 256, batch_mode=2, split=split)
    conv6 = Dropout(0.5)(conv6)
    
    up7 = merge([UpSampling2D(size=(2, 2))(conv6), conv3], mode='concat', concat_axis=1)
    conv7 = inception_block(up7, 128, batch_mode=2, split=split)
    conv7 = Dropout(0.5)(conv7)
    
    up8 = merge([UpSampling2D(size=(2, 2))(conv7), conv2], mode='concat', concat_axis=1)
    conv8 = inception_block(up8, 64, batch_mode=2, split=split)
    conv8 = Dropout(0.5)(conv8)
    
    up9 = merge([UpSampling2D(size=(2, 2))(conv8), conv1], mode='concat', concat_axis=1)
    conv9 = inception_block(up9, 32, batch_mode=2, split=split)
    conv9 = Dropout(0.5)(conv9)
    conv10 = Convolution2D(shp[0], 1, 1, init='he_normal', activation='linear', name='main_output')(conv9) #W_regularizer=l2(0.01)
    
    model = Model(input=inputs, output=[conv10, hc_red_conv3])
    
    if weights_path <> '':
        model.load_weights(weights_path)
    
    model.compile(optimizer=Adam(lr=cfg.learning_rate), loss={'main_output': huber, 'aux_output': huber}, loss_weights={'main_output': 0.8, 'aux_output': 0.2})
    
    return model
