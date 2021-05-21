import os
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
import GeneratorSequence as gs
import saveloadmodel as slm
from statistics import mean
import time

from keras.utils import to_categorical
from keras.utils.generic_utils import get_custom_objects
from keras import backend as K

h = 128
w = 256
hdf_data = h5py.File('..\\data\\HDF_' + str(h) + 'x' + str(w) + '_aug.h5', 'r')

n_classes = 17
batch_size = 16
epochs = 1

# a hirhedt Hargitai féle loss function, source: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
def iou_loss(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
  union = K.sum(y_true, [1, 2, 3])+K.sum(y_pred, [1, 2, 3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return 1 - iou
get_custom_objects().update({"iou_loss": iou_loss}) # defining new loss function for keras


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
            (slm.load_last_model('NET_128X256_CAT13'), [13]),
            (slm.load_last_model('NET_128X256_CAT14'), [14]),
            # (slm.load_last_model('NET_128X256_CAT15'), [15]),
            (slm.load_last_model('NET_128X256_CAT16'), [16])
           ]

labellist = [[70, 70, 70],      # Building      0
             [100, 40, 40],     # Fence->other  1
             [220, 20, 60],     # Pedestrian    2
             [153, 153, 153],   # Pole          3
             [157, 234, 50],    # RoadLine      4
             [128, 64, 128],    # Road          5
             [244, 35, 232],    # Sidewalk      6
             [107, 142, 35],    # Vegetation    7
             [0, 0, 142],       # Vehicles      8
             [102, 102, 156],   # Wall          9
             [220, 220, 0],     # TrafficSign   10
             [70, 130, 180],    # Sky           11
             [81, 0, 81],       # Ground        12
             [230, 150, 140],   # Railtrack     13
             [250, 170, 30],    # TrafficLight  14
             # [110, 190, 160],   # Static        15 -> 1
             # [170, 120, 50],    # Dinamic       16 -> 1
             [45, 60, 150],     # Water         17 -> 15
             [145, 170, 100],    # Terrain       18 -> 16
             [0, 0, 0]
             ]
labels_hu = ['Épület', 'Egyéb', 'Járókelő', 'Villanyoszlop', 'Felfestés', 'Út', 'Járda', 'Növényzet', 'Jármű',\
             'Fal', 'Közúti tábla', 'Égbolt', 'Talaj', 'Sín', 'Közúti lámpa', 'Víz', 'Terep', 'Ismeretlen']

imgs = hdf_data['test'][0:1,:,:,:]

def predict(imgs, threshold = 0.25):
    preds = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2], len(labellist)))
            
    for (m,l) in ml_list:
        preds[:,:,:,l] = m.predict(imgs[:,:,:,0:4])
    
    preds[:,:,:,-1:] = 0.25*np.ones((imgs.shape[0], imgs.shape[1], imgs.shape[2], 1))
    preds = np.argmax(preds, axis = 3)
    preds = to_categorical(preds, num_classes = len(labellist))
    
    return preds

def cat2col(img_cat):
    img_color = np.zeros((img_cat.shape[0],img_cat.shape[1],3), dtype=np.uint8)
    for cindex, col in enumerate(img_cat):
        for lindex, labl in enumerate(col):
            img_color[cindex,lindex,:] = labellist[np.argmax(labl)]
    return img_color


#%% load test data
testsize = 1000
teststart = 0*6+00

imgs = np.array(hdf_data['test'][teststart:teststart+testsize,:,:,:])
preds = predict(imgs)
truth = to_categorical(imgs[:,:,:,4:5], num_classes = len(labellist))

#%% calc IoU metrics
ious_cat = []
for i in range(len(labellist)):
    ious = []
    for j in range(preds.shape[0]):
        ious.append(1-iou_loss(truth[j:j+1,:,:,i:i+1], preds[j:j+1,:,:,i:i+1]).numpy())
    ious_cat.append(ious)

#%% plot IoUs
for i in range(len(labellist)):
    plt.figure()
    plt.title(labels_hu[i])
    plt.plot(ious_cat[i])
    print('cat%02d\t%.4f' % (i, mean(ious_cat[i])))

#%% plot segmentation
for i in range(testsize):
    fig = plt.figure(figsize = (15,8))
    plt.subplot(311)
    plt.title('Nappali teszt', fontsize = 18)
    plt.axis('off')
    plt.imshow(imgs[i,:,:,0:3])
    plt.subplot(312)
    plt.title('Valódi', fontsize = 18)
    plt.axis('off')
    plt.imshow(cat2col(to_categorical(imgs[i,:,:,4:5], num_classes = len(labellist))))
    plt.subplot(313)
    plt.title('Predikció', fontsize = 18)
    plt.axis('off')
    plt.imshow(cat2col(preds[i,:,:,:]))
    
#%% időmérés

preds = np.zeros((imgs.shape[0], imgs.shape[1], imgs.shape[2], len(labellist)))
start_time = time.time()
for i in range(testsize):
    preds[[i]] = predict(imgs[[i]])
    
# preds = predict(imgs)
end_time = time.time()
print('Segmentation for a single frame: %.2f ms' % ((end_time-start_time)*1000/testsize))

