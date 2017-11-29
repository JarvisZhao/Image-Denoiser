
# coding: utf-8

# In[1]:


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
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers import Add
from keras.layers.normalization import BatchNormalization as BN
from keras.regularizers import l2





# In[4]:

class Hyperparameters(object):
    pass

def GenHyperparameters():
    hp = Hyperparameters() 
    hp.depth1 = np.random.choice([4])
    hp.depth2 = np.random.choice([2])
    hp.filters = np.random.choice([32,64])
    hp.nodes = np.random.choice([16,64,128])
    hp.stopping = np.random.choice([5])
    hp.dropout = np.random.choice([0,0.2])
    return hp    


# In[13]:

def bn_relu_conv_block(filters,kernel_size,strides,padding):
    def f(input):
        input = BN()(input)
        input = Activation('relu')(norm)
        return Conv2D(filters = filters,kernel_size=kernel_size,strides=strides,padding=padding)(input)
    return f

def normal_residual_unit(inputs,filter):
    shortcut = inputs
    shortcut = bn_relu_conv_block(filter,3,)

def bottleneck_residual_unit(filter,is_first_block_of_first_layer=False):
    def f(input):

    shortcut = inputs
    shortcut = Conv2D(filters=filter,1,padding='same')(shortcut)
    shortcut = Conv2D(filters=filter,3,padding='same')(shortcut)
    shortcut = Conv2D(filters=filter*4,1,padding='same')(shortcut)
    added = Add()([inputs,shortcut])

    return added



def BuildModel(input_shape,num_outputs,block,structure):
    input = Input(shape = input_shape)
    conv1 = Conv2D(filters=64, kernel_size=(7,7),strides=(2,2))(input)
    pool1 = MaxPooling2D(pool_size=(3,3),strides=(2,2),padding="same")(conv1)
    def BuildModule(inputs,depth=1,filters=16):

        shortcut = inputs
        shortcut = Conv2D(filters,1,padding='same')(shortcut)
        shortcut = Conv2D(filters,3,padding='same')(shortcut)
        shortcut = Conv2D(filters*4,1,padding='same')(shortcut)
        added = Add()()
        out=None
        for k in range(depth):
            
                    
            if k==depth-1:
                #inputs = BatchNormalization()(inputs)
                inputs = Activation('relu')(inputs)
                inputs = Conv2D(filters,3,padding='same')(inputs)
                
                added = Add()([inputs,shortcut])
                out = MaxPooling2D(strides=(2,2))(added)
                
                
            else:
                #inputs = BatchNormalization()(inputs)
                inputs = Activation('relu')(inputs)
            
                inputs = Conv2D(filters,3,padding='same')(inputs)
            
                
        return out

    filters = hp.filters
    input_flag = True
    inputs = Input(shape=(32,32,3))
    out = inputs
    for k in np.arange(hp.depth1)+1:
        out = BuildModule(out,depth=hp.depth2,
                filters=filters)
        input_flag=False
        filters*=2
    out = Flatten()(out)
    out = Dense(hp.nodes)(out)
    out = BatchNormalization()(out)
    out = Activation('relu')(out)
    out = Dropout(hp.dropout)(out)
    out = Dense(10)(out)
    out = Activation('softmax')(out)
    # < code ommitted >
    model = Model(inputs=inputs,outputs=out)
    
    return model


# In[14]:

N      = 50000
EPOCHS = 100
BATCH  = 1000
TRIALS = 10
#already = set()
hp = GenHyperparameters()

model = BuildModel(hp)
cb=TensorBoard(log_dir='Resnet', histogram_freq=0,  
      write_graph=True, write_images=True)
model.compile(loss='categorical_crossentropy',optimizer='Adam',metrics=['accuracy'])
print(model.summary())
time_start = time.time()
hist = model.fit(x_train[:N,:],y_train[:N,:],
    batch_size=BATCH,
    epochs=EPOCHS,
    validation_split=0.2,
    #
    callbacks=[EarlyStopping(patience=hp.stopping),cb]
    )
time_stop = time.time()
time_elapsed = (time_stop - time_start)/60
train_err = 1 - hist.history['acc'][-1]
val_err = 1 - hist.history['val_acc'][-1]
test_acc = model.evaluate(x_test,y_test,batch_size=BATCH,verbose=0)
test_err = 1 - test_acc[1]


