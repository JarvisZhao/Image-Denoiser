
# coding: utf-8

# In[9]:

import matplotlib.pyplot as plt 
import matplotlib.image as mpimg
import numpy as np 
import os


# In[10]:

dir = '../Dataset/ucid.v2.tar'
def read(dir,folder=None):
    if folder is not None:
        new_dir = dir+'/'+folder+'/'
    else:
        new_dir = dir+'/'
    images = list()
    nl = os.listdir(new_dir)
    nl = [x for x in nl if ('noised' not in x and '887' not in x)]
    for name in nl:
        img = mpimg.imread(new_dir+name)
        img= np.array(img)
        if img.shape ==(512,384,3):
            img = img.transpose([1,0,2])
        images.append(img)
    images = np.array(images)
    
    return images


# In[11]:

def read_all_save(dir):
    dirs = os.listdir(dir)
    dirs = [x for x in dirs if 'noised' in x]
    
    original = read(dir)
    tempo = original
    nosied = np.empty([384,512,3]) 
    for k in range(len(dirs)-1):
        original=np.concatenate((original,tempo))
        print(original.shape)
    count = 0
    
    #np.savez('images.npz',original)
    
    noises = list()
    for folder in dirs:
        noise = read(dir,folder)
        print(folder)
        noises.append(noise)
        
        del noise
    np.savez('images.npz',original = original , **{name:value for name, value in zip(dirs,noises)})


# In[12]:

read_all_save(dir)


