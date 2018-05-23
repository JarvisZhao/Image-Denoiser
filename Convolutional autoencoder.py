# This file is training the simple model with ranged relu and use preactivation technique.


import h5py
import numpy as np
import pandas as pd
import time
from keras.models import Model
from keras import backend as K
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
import math
#get_ipython().run_line_magic('matplotlib', 'inline')
#export CUDA_VISIBLE_DEVICES=2,3,4
#jupyter notebook --ip --port --no-browser


# In[3]:


def range_relu(x):
    
    return K.relu(x,max_value=1.00)


# In[5]:


def buildData(n):
    with h5py.File('data.h5', 'r') as hf:
        original = hf['original'][:]
        original1 = original/255
        #if type(n) !=list:
        for k in range(n):
            if k ==0:
                target = original1
            else:
                target = np.concatenate((target,original1),axis = 0)
            #print('target',target.shape)
        
        k = 0
        for key in hf.keys():
            if(k==n):
                break
            if not key in ['original','features']:
                
                noised1 = hf[key][:]
                if(k==0):
                    noised =noised1
                else:
                    noised = np.concatenate((noised,noised1),axis=0)
                k+=1
                #print(noised)
#     with h5py.File('features.h5', 'r') as hf:
#         target = hf['features'][:n*886]
    return target,noised


# In[6]:



N      = 1
EPOCHS = 1000
BATCH  = 64
X = 700

target,noised = buildData(N)


# In[7]:


noised[:X,:].shape


# In[9]:


input_img = Input(shape=(384,512,3))

x = BN()(input_img)
x = Activation(range_relu)(x)
x = Conv2D(16, 3,strides=(2,2),padding='same')(x) #nb_filter, nb_row, nb_col
x = BN()(x)
x = Activation(range_relu)(x)
x = Conv2D(64, 3,strides=(2,2),padding='same')(x)
x = BN()(x)
x = Activation(range_relu)(x)
x = Conv2D(128, 3,strides=(2,2),padding='same')(x)
x = BN()(x)
x = Activation(range_relu)(x)
x = Conv2DTranspose(128, 3,padding='same')(x)
x = UpSampling2D((2,2))(x)
x = BN()(x)
x = Activation(range_relu)(x)
x = Conv2DTranspose(64, 3,padding='same')(x)
x = UpSampling2D((2,2))(x)
#x = UpSampling2D((2,2))(x)
# In original tutorial, border_mode='same' was used. 
# then the shape of 'decoded' will be 32 x 32, instead of 28 x 28
# x = Convolution2D(16, 3, 3, activation='relu', border_mode='same')(x) 
x = BN()(x)
x = Activation(range_relu)(x)
x = Conv2DTranspose(16, 3,padding='same')(x) 
x = UpSampling2D((2,2))(x)
x = BN()(x)
x = Activation(range_relu)(x)
x = Conv2D(3, 5, padding='same')(x)
x = BN()(x)
x = Activation(range_relu)(x)
#x = relu()(x)

model = Model(input_img, x)
gpus = [0,1,3,4,5,6,7]
multi_model = multi_gpu_model(model,gpus= gpus)


# In[10]:



cb=TensorBoard(log_dir='Resnet', histogram_freq=0,  
      write_graph=True, write_images=True)
multi_model.compile(loss='mse',optimizer='Adam')
print(model.summary())


# In[ ]:


hist = multi_model.fit(noised[:X,:],target[:X,:],
    batch_size=BATCH,
    epochs=EPOCHS,
    validation_split=0.2,
    #
    callbacks=[cb,EarlyStopping(patience=100)],
    )
print('Finish Trainging')


# In[ ]:


predicted = multi_model.predict(noised[X:])


# In[18]:


multi_model.save_weights('models/'+'preact_very_very_good.h5')


# In[9]:


# multi_model.load_weights('models/'+'preact_very_very_good.h5')

