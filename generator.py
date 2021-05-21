import numpy as np
import math
import data_augmentation as da
from keras.utils import to_categorical

# yields a batch for neural network
def generator(dataset, indices, n_classes, batch_size = 32, category = -1, isTest = False):
    multiplier = 16
    
    # augmentálást tároló tömb
    augs = np.zeros((len(indices), multiplier), dtype=np.uint)
    for i in range(len(indices)):
        augs[i,:] = np.random.permutation(multiplier)
    augs = augs.flatten('F')
    
    # indexek sokszorozása
    indices = np.repeat([indices], multiplier, axis = 0).flatten()
    
    shape0 = (0, dataset.shape[1], dataset.shape[2], dataset.shape[3])

    if isTest:
        print('indices: ' + str(indices.shape) + 'augs: ' + str(augs.shape))
    
    for i in range(0, len(indices), batch_size):
        # HDF beolvasás osszevissza index esetén
        if type(dataset) == h5py._hl.dataset.Dataset:
            imgs = np.empty(shape0) #, dtype=np.uint8)
            for j in range(batch_size):
                if i+j >= len(indices):
                    break
                imgs = np.append(imgs, dataset[indices[i+j]:indices[i+j]+1], axis=0)
            aug_batch = augs[i:i+imgs.shape[0]]
        
        # numpy fancy indexing
        if type(dataset) == np.ndarray:
            if i+batch_size >= len(indices): # last batch
                aug_batch = augs[i:]
                imgs = np.array(dataset[indices[i:],:,:,0:5]) #, dtype=np.uint8)
            else:
                aug_batch = augs[i:i+batch_size]
                imgs = np.array(dataset[indices[i:i+batch_size],:,:,0:5]) #, dtype=np.uint8)
            
        imgs = np.concatenate((imgs[:,:,:,0:4], to_categorical(imgs[:,:,:,4:5], num_classes = n_classes)), axis = 3) #, dtype=np.uint8)), axis = 3)
            
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
           
        if category != -1:
            Y_batch = Y_batch[:,:,:,category:category+1] # itt csak az egy kiválaszott kategóriát adjuk tovább
            
        if isTest:
             yield (X_batch, Y_batch), aug_batch
             
        yield (X_batch, Y_batch)