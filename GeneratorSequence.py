import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import KFold
import data_augmentation as da
import time

from keras.utils import to_categorical
import keras

class GeneratorSequence(keras.utils.Sequence):
    def __init__(self, dataset, indices, n_classes, batch_size = 32, categories = [], category = -1, isTest = False):
        self.dataset = dataset
        self.n_classes = n_classes
        self.batch_size = batch_size
        self.category = category
        self.categories = categories
        self.isTest = isTest
        self.multiplier = 16
    
        # augmentálást tároló tömb
        augs = np.zeros((len(indices), self.multiplier), dtype=np.uint)
        for i in range(len(indices)):
            augs[i,:] = np.random.permutation(self.multiplier)
        self.augs = augs.flatten('F')
        
        # indexek sokszorozása
        self.indices = np.repeat([indices], self.multiplier, axis = 0).flatten()
        
        self.shape0 = (0, dataset.shape[1], dataset.shape[2], dataset.shape[3])
        if self.isTest:
            print('indices: ' + str(self.indices.shape) + 'augs: ' + str(self.augs.shape))

    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))
    
    def __getitem__(self, index):
        if (index+1)*self.batch_size > len(self.indices):
            ii = list(range(index*self.batch_size, len(self.indices)))
        else:
            ii = list(range(index*self.batch_size, (index+1)*self.batch_size))
            
        # generate data
        return self.__data_generation(ii)

        
    def __data_generation(self, ii):    
        # HDF beolvasás osszevissza index esetén
        if type(self.dataset) == h5py._hl.dataset.Dataset:
            imgs = np.empty(self.shape0) #, dtype=np.uint8)
            for j in ii:
                imgs = np.append(imgs, self.dataset[self.indices[j]:self.indices[j]+1], axis=0)
            aug_batch = self.augs[ii]
        
        # numpy fancy indexing
        if type(self.dataset) == np.ndarray:
            aug_batch = self.augs[ii]
            imgs = np.array(self.dataset[self.indices[ii],:,:,0:5]) #, dtype=np.uint8)
            
        imgs = np.concatenate((imgs[:,:,:,0:4], to_categorical(imgs[:,:,:,4:5], num_classes = self.n_classes)), axis = 3) #, dtype=np.uint8)), axis = 3)
            
        aug_f = [x for x in range(len(aug_batch)) if aug_batch[x] & 0b0001]
        aug_b = [x for x in range(len(aug_batch)) if aug_batch[x] & 0b0010]
        aug_g = [x for x in range(len(aug_batch)) if aug_batch[x] & 0b0100]
        aug_r = [x for x in range(len(aug_batch)) if aug_batch[x] & 0b1000]
        
        imgs[aug_f] = da.aug_flip(imgs[aug_f])
        imgs[aug_b] = da.aug_brightness(imgs[aug_b], [np.random.randint(-50,-20), np.random.randint(20,50)][np.random.randint(0,2)])
        imgs[aug_g] = da.aug_gauss_noise(imgs[aug_g], 25)
        imgs[aug_r] = da.aug_rect(imgs[aug_r], 0.05 + np.random.rand()/5, 0.05 + np.random.rand()/5, np.random.rand(), np.random.rand())
        imgs[aug_r] = da.aug_rect(imgs[aug_r], 0.05 + np.random.rand()/5, 0.05 + np.random.rand()/5, np.random.rand(), np.random.rand())
        imgs[aug_r] = da.aug_rect(imgs[aug_r], 0.05 + np.random.rand()/5, 0.05 + np.random.rand()/5, np.random.rand(), np.random.rand())
        
        X_batch = imgs[:,:,:,0:4]
        Y_batch = imgs[:,:,:,4:]
        
        if len(self.categories) != 0:
            Y_batch = Y_batch[:,:,:,self.categories]
        elif self.category != -1:
            Y_batch = Y_batch[:,:,:,self.category:self.category+1] # itt csak az egy kiválaszott kategóriát adjuk tovább
            
        if self.isTest:
            return (X_batch, Y_batch), aug_batch
        return (X_batch, Y_batch)
    
class GeneratorConet(keras.utils.Sequence):
    
    def __init__(self, dataset, indices, n_classes, model_and_layer_list, batch_size = 32, isTest = False):
        self.dataset = dataset
        self.indices = np.array(indices, dtype=np.int)
        self.n_classes = n_classes
        self.ml_list = model_and_layer_list
        self.batch_size = batch_size
        self.isTest = isTest
        self.im_shape0 = (0, dataset.shape[1], dataset.shape[2], dataset.shape[3])
        
        
    def __len__(self):
        return int(np.floor(len(self.indices) / self.batch_size))
        
    def __getitem__(self, index):
        if (index+1)*self.batch_size > len(self.indices):
            ii = list(range(index*self.batch_size, len(self.indices)))
        else:
            ii = list(range(index*self.batch_size, (index+1)*self.batch_size))
            
        # generate data
        return self.__data_generation(ii)
            
    
    def __data_generation(self, ii):    
        # HDF beolvasás osszevissza index esetén
        if type(self.dataset) == h5py._hl.dataset.Dataset:
            imgs = np.empty(self.im_shape0) #, dtype=np.uint8)
            for j in ii:
                imgs = np.append(imgs, self.dataset[self.indices[j]:self.indices[j]+1], axis=0)
        
        # numpy fancy indexing
        if type(self.dataset) == np.ndarray:
            imgs = np.array(self.dataset[self.indices[ii],:,:,0:5]) #, dtype=np.uint8)
            
        X_batch = np.zeros((len(ii), self.dataset.shape[1], self.dataset.shape[2], self.n_classes))
        
        for (m,l) in self.ml_list:
            # if self.isTest:
            #     m.summary()
            #     print('try to pred shape: ', imgs[:,:,:,0:4].shape)
            #     print('list: ', l)
            X_batch[:,:,:,l] = m.predict(imgs[:,:,:,0:4])
        
        Y_batch = to_categorical(imgs[:,:,:,4:5], num_classes = self.n_classes)
        
        return (X_batch, Y_batch)
            