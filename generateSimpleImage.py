import numpy as np
import h5py
import matplotlib.pyplot as plt


class parameters(object):
    pass

def GenHyperparameters():
    hp = parameters() 
    hp.color = np.random.choice([1,3])
    hp.boundary = np.random.choice(['clear','blur'])
    hp.orientation = np.random.choice(['horizental','vertical'])
    return hp


hp = GenHyperparameters()
def buildImage(hp):
    img = np.zeros([128,128,3])
    
    color1 = np.random.rand(hp.color)
    color2 = np.random.rand(hp.color)
    if hp.orientation == 'horizental':
        pos = np.random.randint(128)
        #
        #print(pos)
        img[:,:pos,:] = color1
        img[:,pos+1:,:] = color2
    elif hp.orientation =='vertical':
        
        pos = np.random.randint(128)
        
        img[:pos,:,:] = color1
        img[pos+1:,:,:] = color2
    return img


imgs = []
for k in range(10000):
    hp = GenHyperparameters()
    img = buildImage(hp)
    imgs.append(img)
imgs = np.array(imgs)
h5 = h5py.File('smallSimpleImages.h5','w')
h5.create_dataset('clean',data = imgs)
h5.close()
