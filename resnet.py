
# coding: utf-8

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

from keras.callbacks import TensorBoard
import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping 
import os 


# In[4]:

file = np.load('images.npz')
original = file['original']
noised1 = file['noised-1']
original = original[:886]
# noised1 = noised1/255
noised1 = noised1[:50]
original = original[:50]
original = original/255
print('orignial ',original.shape)
print('noised   ',noised1.shape)


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
    block = bn_relu_deconv_block(filters=filters,kernel_size=2,strides=(2,2))(block)
    block = Conv2DTranspose(filters=3,kernel_size=1,strides=(2,2))(block)
    
    
    #block = Flatten()(block)
    # for k in f_c:
    #     block = Dense(k)(block)
    # block = Dense(384*384*3)(block)
    # < code ommitted >
    model = Model(inputs=input,outputs=block)
    
    return model


# In[14]:

N      = 50
EPOCHS = 5
BATCH  = 50
TRIALS = 10


structure = [2,2,2,2]
model = BuildModel(input_shape=[384,512,3],init_filters=64,blockfn='basic',structure=structure)
cb=TensorBoard(log_dir='Resnet', histogram_freq=0,  
      write_graph=True, write_images=True)
model.compile(loss='mean_squared_error',optimizer='Adam')
print(model.summary())
time_start = time.time()
hist = model.fit(noised1[:N,:],original[:N,:],
    batch_size=BATCH,
    epochs=EPOCHS,
    validation_split=0.2,
    #
    callbacks=[EarlyStopping(patience=10),cb]
    )
name = str(structure)+'.h5'
model.save_weights('models/'+name)
time_stop = time.time()
time_elapsed = (time_stop - time_start)/60

predicted = model.predict(noised)
plt.imshow(predicted[0])


# train_err = 1 - hist.history['acc'][-1]
# val_err = 1 - hist.history['val_acc'][-1]
# test_acc = model.evaluate(x_test,y_test,batch_size=BATCH,verbose=0)
# test_err = 1 - test_acc[1]


