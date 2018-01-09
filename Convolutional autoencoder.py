import h5py
import numpy as np
import pandas as pd
import time
from keras.models import Model
from keras.layers import (
    Input,
    Activation,
    Dense,
    Flatten,
    Reshape,
    UpSampling2D
)
from keras.layers.convolutional import (
    Conv2D,
    Conv2DTranspose,
    MaxPooling2D,
    AveragePooling2D,
    #UpSampleing2D
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
from training_utils import multi_gpu_model


file = np.load('images.npz')
original = file['original']
noised1 = file['noised-1']
original = original[:886]
# noised1 = noised1/255

original = original/255
print('orignial ',original.shape)
print('noised   ',noised1.shape)

input_img = Input(shape=(384,512,3))

x = Conv2D(16, 3, 3, activation='relu', border_mode='same')(input_img) #nb_filter, nb_row, nb_col
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = MaxPooling2D((2, 2), border_mode='same')(x)
x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(x)
encoded = MaxPooling2D((2, 2), border_mode='same')(x)

x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(encoded)
x = UpSampling2D((2, 2))(x)
x = Conv2D(8, 3, 3, activation='relu', border_mode='same')(x)
x = UpSampling2D((2, 2))(x)

# In original tutorial, border_mode='same' was used. 
# then the shape of 'decoded' will be 32 x 32, instead of 28 x 28
# x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x) 
x = Conv2D(16, 3, 3, activation='relu', border_mode='same')(x) 

x = UpSampling2D((2, 2))(x)
decoded = Conv2D(1, 5, 5, activation='sigmoid', border_mode='same')(x)

model = Model(input_img, decoded)

N      = 886
EPOCHS = 10
BATCH  = 50
TRIALS = 10


cb=TensorBoard(log_dir='Resnet', histogram_freq=0,  
      write_graph=True, write_images=True)
model.compile(loss='mean_squared_error',optimizer='Adam')
print(model.summary())