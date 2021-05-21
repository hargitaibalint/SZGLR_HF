import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
import time

from keras.utils import to_categorical
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects

from generator import generator
from GeneratorSequence import GeneratorSequence, GeneratorConet
import saveloadmodel as slm

# a hirhedt Hargitai féle loss function, source: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
def iou_loss(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
  union = K.sum(y_true, [1, 2, 3])+K.sum(y_pred, [1, 2, 3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return 1 - iou
get_custom_objects().update({"iou_loss": iou_loss}) # defining new loss function for keras
        
#%% test_this_shit
n_classes = 17
batch_size = 16

h = 128
w = 256
hdf_data = h5py.File('..\\data\\HDF_' + str(h) + 'x' + str(w) + '_aug.h5', 'r')

indicies = list(range(hdf_data['train_valid'].shape[0]))
# np.random.shuffle(indicies)

train_valid = np.array(hdf_data['train_valid'], dtype = np.uint8)

# train_gen = generator(train_valid, indicies, n_classes, batch_size = batch_size, isTest = True)
train_gen = GeneratorSequence(train_valid, indicies, n_classes, batch_size = batch_size, categories= [0,5,11], isTest = True)

X_from_generator = np.zeros((1024,h,w,4)) #, dtype=np.uint8)
Y_from_generator = np.zeros((1024,h,w,n_classes)) #, dtype=np.uint8)
augs_from_generator = np.zeros((1024), dtype=np.uint8)
#%%

start_time = time.time()
ind = 0
for (X,Y), augs in train_gen:
    if(ind+batch_size <= 1024 and np.random.rand() > 0):
        X_from_generator[ind:ind+batch_size] = X
        # Y_from_generator[ind:ind+batch_size] = Y
        augs_from_generator[ind:ind+batch_size] = augs
        print(ind)
        ind = ind+batch_size
end_time = time.time()
print('Reading the whole generator in %d batches is: %dsecs' %(batch_size, end_time-start_time))

#%%
for i in range(0, ind, 16):
    x = np.array(X_from_generator[i], dtype=np.uint8)
    # y = np.array(Y_from_generator[i], dtype=np.uint8)
    a = augs_from_generator[i]
    
    title = ''
    if a & 0b0001:
        title += 'Tükrözve'
    if a & 0b0010:
        if title != '':
            title += ' + '
        title += 'Fényerő'
    if a & 0b0100:
        if title != '':
            title += ' + '
        title += 'Gauss zaj'
    if a & 0b1000:
        if title != '':
            title += ' + '
        title += "Téglalapok "
        
    if title == '':
        title = 'Eredeti'
        
    fig = plt.figure(figsize=(9,4.2))
    plt.axis('off')
    fig.suptitle(title, fontsize=20, fontweight="bold")
    plt.imshow(x[:,:,0:3])
        
    # y = np.argmax(y, axis=2)
        
    # fig = plt.figure()
    # fig.suptitle(title)
    # plt.subplot(311)
    # plt.imshow(x[:,:,0:3])
    # plt.subplot(312)
    # plt.imshow(x[:,:,3:4], cmap='gray', vmin=0, vmax=255)
    # plt.subplot(313)
    # plt.imshow(y*10, cmap='gray', vmin=0, vmax=255)
    
#%%
(X,Y), augs = train_gen[0]
for i in range(0, X.shape[0], 1):
    x = np.array(X[i], dtype=np.uint8)
    y = np.array(Y[i], dtype=np.uint8)
    print(y.shape)
    
    a = augs[i]
    title = ''
    if a & 0b0001:
        title += 'Tükrözve'
    if a & 0b0010:
        if title != '':
            title += ' + '
        title += 'Fényerő'
    if a & 0b0100:
        if title != '':
            title += ' + '
        title += 'Gauss zaj'
    if a & 0b1000:
        if title != '':
            title += ' + '
        title += "Téglalapok "
        
    if title == '':
        title = 'Eredeti'
        
    y = np.argmax(y, axis=2)
        
    fig = plt.figure(figsize=(6,2.8))
    plt.axis('off')
    fig.suptitle(title)
    plt.imshow(x[:,:,0:3])
    # plt.subplot(311)
    # plt.imshow(x[:,:,0:3])
    # plt.subplot(312)
    # plt.imshow(x[:,:,3:4], cmap='gray', vmin=0, vmax=255)
    # plt.subplot(313)
    # plt.imshow(y*(255//np.max(y)), cmap='gray', vmin=0, vmax=255)
    

#%%
ml_list = [
            (slm.load_last_model('NET_128X256_CAT00_05_11'), [0,5,11]),
            (slm.load_last_model('NET_128X256_CAT01'), [1]),
            (slm.load_last_model('NET_128X256_CAT02'), [2]),
            (slm.load_last_model('NET_128X256_CAT03'), [3]),
            (slm.load_last_model('NET_128X256_CAT04'), [4]),
            (slm.load_last_model('NET_128X256_CAT06'), [6]),
            (slm.load_last_model('NET_128X256_CAT07'), [7]),
            (slm.load_last_model('NET_128X256_CAT08'), [8]),
            (slm.load_last_model('NET_128X256_CAT09'), [9]),
            # (slm.load_last_model('NET_128X256_CAT10'), [10]),
            (slm.load_last_model('NET_128X256_CAT12'), [12]),
            # (slm.load_last_model('NET_128X256_CAT13'), [13]),
            (slm.load_last_model('NET_128X256_CAT14'), [14]),
            # (slm.load_last_model('NET_128X256_CAT15'), [15]),
            (slm.load_last_model('NET_128X256_CAT16'), [16])
           ]
#%%
train_gen = GeneratorConet(train_valid, indicies, n_classes, ml_list, batch_size = batch_size)

#%%
start_time = time.time()
ind = 0
for (X,Y) in train_gen:
    if(np.random.rand() > 0.98):
        print('Szerencsés két százalék')
end_time = time.time()
print('Reading the whole generator in %d batches is: %dsecs' %(batch_size, end_time-start_time))
#%%
(X,Y) = train_gen[0]

for i in range(0, 4, 1):
    for j in range(0, X.shape[-1], 1):
 
        fig = plt.figure()
        fig.suptitle('%d.kép, %d.kat' % (i,j))
        plt.subplot(211)
        plt.imshow(X[i,:,:,j:j+1], cmap='gray', vmin=0, vmax=1)
        plt.subplot(212)
        plt.imshow(Y[i,:,:,j:j+1], cmap='gray', vmin=0, vmax=1)