
# coding: utf-8

# In[1]:


import numpy as np
import pandas as pd
import time
from keras.utils import np_utils
from keras.models import Model
from keras.models import Sequential
from keras.layers.core import Dense,Activation,Dropout,Flatten
from keras.layers import Input
from keras.layers.convolutional import (
    Conv2D,
    MaxPooling2D,
    AveragePooling2D
)
from keras.layers.normalization import BatchNormalization
from keras.optimizers import Adam
from keras.callbacks import EarlyStopping,TensorBoard

from keras.layers import Add


# In[2]:



# load CIFAR10 dataset
print('Loading CIFAR-10 dataset...')
CIFAR10 = np.load('CIFAR-10_train.npz')
x_train = CIFAR10['train_images']
y_train = CIFAR10['train_labels']
CIFAR10 = np.load('CIFAR-10_test.npz')
x_test  = CIFAR10['test_images']
y_test  = CIFAR10['test_labels']


# In[3]:

# preprocess CIFAR10 data
print('Preprocessing CIFAR10 dataset...')
x_train = x_train.astype('float32')
x_test  = x_test.astype('float32')
x_train = (x_train - x_train.mean())/x_train.std()
x_test  = (x_test - x_test.mean())/x_test.std()
y_train = np_utils.to_categorical(y_train,10)
y_test  = np_utils.to_categorical(y_test,10)
print('    train images tensor:',x_train.shape)
print('    test  images tensor:',x_test.shape)
print('    train labels tensor:',y_train.shape)
print('    test  labels tensor:',y_test.shape)
print()


# In[ ]:




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

def BuildModel(hp):
    
    def BuildModule(inputs,depth=1,filters=16):

        shortcut = inputs
        shortcut = Conv2D(filters,1,padding='same')(shortcut)
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
already = set()
cols = ['depth1','depth2','filters','nodes','parameters','stopping','dropout','epochs',
    'time (min)','train error','val error','test error']
df = pd.DataFrame(np.zeros((TRIALS,len(cols))).fill(np.nan),columns=cols)
for trial in range(TRIALS): 
    print('trial = %d/%d' % (trial+1,TRIALS))
    hp = GenHyperparameters()
    while hp in already:
    	hp = GenHyperparameters()
    already.add(hp)
    df.loc[trial,'depth1']   = hp.depth1
    df.loc[trial,'depth2']   = hp.depth2
    df.loc[trial,'filters']  = hp.filters
    df.loc[trial,'nodes']    = hp.nodes
    df.loc[trial,'stopping'] = hp.stopping 
    df.loc[trial,'dropout']  = hp.dropout
    df = df.sort_values('val error')
    df.to_csv('CIFAR10_CNN2.csv',index=False)

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
    df.loc[trial,'parameters']  = model.count_params()
    df.loc[trial,'epochs']      = hist.epoch[-1]
    df.loc[trial,'time (min)']  = time_elapsed
    df.loc[trial,'train error'] = train_err 
    df.loc[trial,'val error']   = val_err 
    df.loc[trial,'test error']  = test_err

    df = df.sort_values('val error')
    df.to_csv('CIFAR10_CNN2.csv',index=False)
    print(df.head().round(2))
    print()


# In[ ]:



