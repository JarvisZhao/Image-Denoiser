
# coding: utf-8
## Run on server
#import tensorflow as tf         
# from keras import backend as K  # needed for mixing TensorFlow and Keras commands 
# config = tf.ConfigProto()
# config.gpu_options.per_process_gpu_memory_fraction = 0.9
# sess = tf.Session(config=config)
# K.set_session(sess)
# In[1]:

import h5py
import numpy as np
import pandas as pd
import time
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten
)
from keras.layers.convolutional import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers import Add
from keras.layers.normalization import BatchNormalization as BN
from keras.regularizers import l2





#original = original/255



# In[13]:

def bn_relu_conv_block(filters,kernel_size,strides=(1,1),padding='same'):
    def f(input):
        input = BN()(input)
        input = Activation('relu')(input)
        return Conv2D(filters = filters,kernel_size=kernel_size,strides=strides,padding=padding)(input)
    return f

def normal_residual_unit(filter,strides=(1,1),is_first_block_of_first_layer=False):
    def f(input):

        shortcut = input
        if is_first_block_of_first_layer:
            conv1 = Conv2D(filters=filter,kernel_size=(3,3),strides=(1,1),padding='same' )(input)
        else:
            conv1 = bn_relu_conv_block(filters=filter,kernel_size=(3,3))(input)
        conv2 = bn_relu_conv_block(filters = filter,kernel_size=(3,3),strides=strides)(conv1)
        shortcut = Conv2D(filters=filter,kernel_size=(1,1),strides=strides,padding='same')(shortcut)
        added = Add()([conv2,shortcut])
        return added
    return f

def bottleneck_residual_unit(filter,strides=(1,1),is_first_block_of_first_layer=False):
    def f(input):
        shortcut = input
        if is_first_block_of_first_layer:
            conv_1_1 = Conv2D(filters = filter,kernel_size=1,strides=(1,1),padding='same')(input)
        else:
            conv_1_1 = bn_relu_conv_block(filters = filter,kernel_size=(1,1),strides=(1,1),padding='same')(input)
        conv_3_3 =  bn_relu_conv_block(filters = filter,kernel_size=(3,3))(conv_1_1)
        res = bn_relu_conv_block(filters=filter*4,kernel_size=(1,1),strides=strides)(conv_3_3)
        shortcut = Conv2D(filters = filter*4,kernel_size=1,strides=strides)(shortcut)
        added = Add()([res,shortcut])
        return added
    return f


def bn_relu_deconv_block(filters,kernel_size,strides=(1,1),padding='same'):
    def f(input):
        input = BN()(input)
        input = Activation('relu')(input)
        return Conv2DTranspose(filters = filters,kernel_size=kernel_size,strides=strides,padding='same')(input)
    return f

def de_normal_residual_unit(filter,strides=(1,1),is_first_block_of_first_layer=False):
    def f(input):

        shortcut = input
        if is_first_block_of_first_layer:
            conv1 = Conv2DTransposeTranspose(filters=filter,kernel_size=(3,3),strides=(1,1),padding='same' )(input)
        else:
            conv1 = bn_relu_deconv_block(filters=filter,kernel_size=(3,3))(input)
        conv2 = bn_relu_deconv_block(filters = filter,kernel_size=(3,3),strides=strides)(conv1)
        shortcut = Conv2DTranspose(filters=filter,kernel_size=(1,1),strides=strides)(shortcut)
        added = Add()([conv2,shortcut])
        return added
    return f

def de_bottleneck_residual_unit(filter,strides=(1,1),is_first_block_of_first_layer=False):
    def f(input):
        shortcut = input
        if is_first_block_of_first_layer:
            conv_1_1 = Conv2DTranspose(filters = filter,kernel_size=1,strides=(1,1),padding='same')(input)
        else:
            conv_1_1 = bn_relu_deconv_block(filters = filter,kernel_size=(1,1),strides=(1,1),padding='same')(input)
        conv_3_3 =  bn_relu_deconv_block(filters = filter,kernel_size=(3,3))(conv_1_1)
        res = bn_relu_deconv_block(filters=filter*4,kernel_size=(1,1),strides=strides)(conv_3_3)
        shortcut = Conv2DTranspose(filters = filter*4,kernel_size=1,strides=strides)(shortcut)
        added = Add()([res,shortcut])
        return added
    return f





def BuildModel(input_shape,init_filters,blockfn,structure):
    input = Input(shape = input_shape)
    conv1 = Conv2D(filters=init_filters, kernel_size=(7,7),strides=(2,2))(input)
    conv1_bn = BN()(conv1)
    conv1_bn_relu = Activation('relu')(conv1_bn)
    pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same")(conv1_bn_relu)
    filters = init_filters
    block = pool1
    print(structure)
    for n in range(len(structure)):
        for k in range(structure[n]):
            if not k==structure[n]-1:
                if blockfn == 'basic':
                    block = normal_residual_unit(filter=filters,strides=(1,1),is_first_block_of_first_layer=(n==0 and k==0))(block)
                elif blockfn == 'bottleneck':
                    block = bottleneck_residual_unit(filter=filters,strides=(1,1),is_first_block_of_first_layer=(n==0 and k==0))(block)
            else:
                if blockfn == 'basic':
                    block = normal_residual_unit(filter=filters,strides=(2,2),is_first_block_of_first_layer=(n==0 and k==0))(block)
                elif blockfn == 'bottleneck':
                    block = bottleneck_residual_unit(filter=filters,strides=(2,2),is_first_block_of_first_layer=(n==0 and k==0))(block)
        filters *= 2
    
    for n in range(len(structure)-1,-1,-1):
        for k in range(structure[n]):
            if not k== structure[n]-1:
                if blockfn == 'basic':
                    block = de_normal_residual_unit(filters)(block)
                elif blockfn == 'bottleneck':
                    block = de_bottleneck_residual_unit(filters)(block)
            else:
                if blockfn == 'basic':
                    block = de_normal_residual_unit(filters,strides=(2,2))(block)
                elif blockfn == 'bottleneck':
                    block = de_bottleneck_residual_unit(filters,strides=(2,2))(block)
        #block = Conv2DTranspose(filters=filters,kernel_size=3,strides=(2,2))(block)
        filters//=2
    block = de_normal_residual_unit(filter=filters,strides=(2,2))(block)
    block = de_normal_residual_unit(filter=3,strides=(2,2))(block)
    block = normal_residual_unit(filter=3,strides=(1,1))(block)
    block = Activation('relu')(block)
    #block = Flatten()(block)
    # for k in f_c:
    #     block = Dense(k)(block)
    # block = Dense(384*384*3)(block)
    # < code ommitted >
    model = Model(inputs=input,outputs=block)
    
    return model