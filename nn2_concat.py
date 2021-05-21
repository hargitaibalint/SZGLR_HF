import os
import numpy as np
import h5py
import math
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from sklearn.model_selection import KFold
import GeneratorSequence as gs
import saveloadmodel as slm

from keras.callbacks import ModelCheckpoint, CSVLogger, Callback
from keras.utils import to_categorical
from keras.models import Sequential
from keras.layers import Conv2D, UpSampling2D, MaxPooling2D
from keras.layers.core import Activation, Dense, Flatten
from keras.optimizers import SGD
from keras.models import load_model, save_model
from keras.layers.convolutional import Conv1D, MaxPooling1D
from keras.models import Model
from keras.layers import Input, merge, ZeroPadding2D, concatenate, BatchNormalization
from keras.layers.core import Dropout
from keras import backend as K
from keras.utils.generic_utils import get_custom_objects
from keras.callbacks import CSVLogger
import tensorflow as tf
tf.get_logger().setLevel('INFO')

h = 128
w = 256
hdf_data = h5py.File('..\\data\\HDF_' + str(h) + 'x' + str(w) + '_aug.h5', 'r')

n_classes = 17
batch_size = 16
epochs = 1

model_name = 'NET_' + str(h) + 'x' + str(w) + '_CONET'
log_path = '..\\logs\\'  + model_name + '.csv'
open(log_path, 'a+').close()    # file letrehozasa ha nem letezne

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
            # (slm.load_last_model('NET_128X256_CAT13'), [13]),
            (slm.load_last_model('NET_128X256_CAT14'), [14]),
            # (slm.load_last_model('NET_128X256_CAT15'), [15]),
            (slm.load_last_model('NET_128X256_CAT16'), [16])
           ]

def create_empty_conet(n_classes):
    inputs = Input((h, w, n_classes))
    conv1 = Conv2D(n_classes, (5, 5), padding='same')(inputs)
    conv1 = BatchNormalization(axis=3, momentum=0.9)(conv1)
    output = Activation('softmax')(conv1)
    # maxpooling
    # dropout
    
    # conv2 = Conv2D(n_classes, (30, 30), padding='same')(conv1)
    # output = Activation('softmax')(conv2)
    
    conet = Model(inputs, output)
    
    conet.compile(optimizer='adam', loss=iou_loss)
    return conet


np_train_valid_dataset = np.array(hdf_data['train_valid'], dtype=np.uint8)

n_samples_list = list(range(hdf_data['train_valid'].shape[0]))

kf = KFold(n_splits=5, shuffle=True)

#%%
model = slm.load_last_model(model_name)
if model == None:
    model = create_empty_conet(n_classes)
    # model.summary()
    my_weights = np.zeros((5, 5, 17, 17))
    for i in range(17):
        my_weights[2,2,i,i] = 1
    model.layers[1].set_weights([my_weights, np.zeros((17))])
    

for i in range(epochs):        
    for train_index, valid_index in kf.split(n_samples_list):
        train_gen = gs.GeneratorConet(np_train_valid_dataset, train_index, n_classes, ml_list, batch_size = batch_size)
        valid_gen = gs.GeneratorConet(np_train_valid_dataset, valid_index, n_classes, ml_list, batch_size = batch_size)
        
        model.fit(train_gen, validation_data = valid_gen, \
                 callbacks = [slm.CheckPointer(model_name), \
                              CSVLogger(log_path, append=True, separator=',')])
            
#%%
testsize = 100
teststart = 00*6+00
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
             [145, 170, 100]    # Terrain       18 -> 16
             ]


test_gen = gs.GeneratorConet(hdf_data['test'], list(range(teststart, teststart + testsize)) , n_classes, ml_list, batch_size = testsize)
(test_x, test_y) = test_gen[0]
# test_y = hdf_data['test'][teststart:teststart+testsize,:,:,4:5]
preds = model.predict(test_x)

#TODO: np.argmax

def cat2col(img_cat):
    img_color = np.zeros((img_cat.shape[0],img_cat.shape[1],3), dtype=np.uint8)
    for cindex, col in enumerate(img_cat):
        for lindex, labl in enumerate(col):
            img_color[cindex,lindex,:] = labellist[np.argmax(labl)]
    return img_color


for i in range(testsize):
    pred = cat2col(preds[i])
    test = cat2col(test_y[i])
    dummy = cat2col(test_x[i])
    
    fig = plt.figure()
    fig.suptitle('%d.kép' % (i))
    plt.subplot(311)
    plt.imshow(pred)
    plt.subplot(312)
    plt.imshow(test)
    plt.subplot(313)
    plt.imshow(dummy)
    
    
my_weights = np.zeros((5, 5, 17, 17))
for i in range(17):
    my_weights[2,2,i,i] = 1


# #%%
# for i in range(19):
#     plt.figure()
#     plt.imshow(preds[0,:,:,i])
#     plt.title(str(i))