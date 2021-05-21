# -*- coding: utf-8 -*-

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

#%%
h = 128
w = 256
hdf_data = h5py.File('..\\data\\HDF_' + str(h) + 'x' + str(w) + '_aug.h5', 'r')

categories = [[8]]

n_classes = 17
batch_size = 64
epochs = 6


# a hirhedt Hargitai féle loss function, source: https://towardsdatascience.com/metrics-to-evaluate-your-semantic-segmentation-model-6bcb99639aa2
def iou_loss(y_true, y_pred, smooth=1):
  intersection = K.sum(K.abs(y_true * y_pred), axis=[1, 2, 3])
  union = K.sum(y_true, [1, 2, 3])+K.sum(y_pred, [1, 2, 3])-intersection
  iou = K.mean((intersection + smooth) / (union + smooth), axis=0)
  return 1 - iou
get_custom_objects().update({"iou_loss": iou_loss}) # defining new loss function for keras


# creating an Unet
# more about the network can be found in our documentation

# TODO: zero padding?

def create_empty_unet(output_layers = 1):
    inputs = Input((h, w, 4))
    # encoder part
    conv1 = Conv2D(8, (3, 3), padding='same')(inputs)
    conv1 = BatchNormalization(axis=3, momentum=0.9)(conv1)
    conv1 = Activation('relu')(conv1)
    # maxpooling
    # dropout
    
    conv2 = Conv2D(16, (3, 3), padding='same', strides=2)(conv1)
    conv2 = BatchNormalization(axis=3, momentum=0.9)(conv2)
    conv2 = Activation('relu')(conv2)
    
    conv3 = Conv2D(32, (3, 3), padding='same', strides=2)(conv2)
    conv3 = BatchNormalization(axis=3, momentum=0.9)(conv3)
    conv3 = Activation('relu')(conv3)
    
    conv4 = Conv2D(64, (3, 3), padding='same', strides=2)(conv3)
    conv4 = BatchNormalization(axis=3, momentum=0.9)(conv4)
    conv4 = Activation('relu')(conv4)
    
    conv5 = Conv2D(128, (3, 3), padding='same', strides=2)(conv4)
    conv5 = BatchNormalization(axis=3, momentum=0.9)(conv5)
    conv5 = Activation('relu')(conv5)
    
    # decoder part
    concat1 = concatenate([UpSampling2D((2, 2))(conv5), conv4], axis=3)
    conv6 = Conv2D(64, (3, 3), padding='same')(concat1)
    conv6 = BatchNormalization(axis=3, momentum=0.9)(conv6)
    conv6 = Activation('relu')(conv6)
    
    concat2 = concatenate([UpSampling2D((2, 2))(conv6), conv3], axis=3)
    conv7 = Conv2D(32, (3, 3), padding='same')(concat2)
    conv7 = BatchNormalization(axis=3, momentum=0.9)(conv7)
    conv7 = Activation('relu')(conv7)
    
    concat3 = concatenate([UpSampling2D((2, 2))(conv7), conv2], axis=3)
    conv8 = Conv2D(16, (3, 3), padding='same')(concat3)
    conv8 = BatchNormalization(axis=3, momentum=0.9)(conv8)
    conv8 = Activation('relu')(conv8)
    
    concat4 = concatenate([UpSampling2D((2, 2))(conv8), conv1], axis=3)
    conv9 = Conv2D(8, (3, 3), padding='same')(concat4)
    conv9 = BatchNormalization(axis=3, momentum=0.9)(conv9)
    conv9 = Activation('relu')(conv9)
    
    output = Conv2D(output_layers, (1, 1), activation='sigmoid', padding='same')(conv9)
    
    unet = Model(inputs, output)
    
    # we used mean absolute error at the beginning
    unet.compile(optimizer='adam', loss=iou_loss)
    
    return unet


# %%

np_train_valid_dataset = np.array(hdf_data['train_valid'], dtype=np.uint8)

n_samples_list = list(range(hdf_data['train_valid'].shape[0]))

kf = KFold(n_splits=5, shuffle=True)

for i in range(epochs):
    for cats in categories:
        model_name = 'NET_' + str(h) + 'x' + str(w) + '_cat%02d' % (cats[0])
        for i in range(1,len(cats)):
            model_name = model_name + '_%02d' % cats[i]
        log_path = '..\\logs\\'  + model_name + '.csv'
        open(log_path, 'a+').close() # file letrehozasa ha nem letezne
        
        model = slm.load_last_model(model_name)
        if model == None:
            model = create_empty_unet(len(cats))
            model.summary()
            
        for train_index, valid_index in kf.split(n_samples_list):
            # train_gen = gs.GeneratorSequence(hdf_data['train_valid'], train_index, n_classes, batch_size = batch_size, categories = cats)
            # valid_gen = gs.GeneratorSequence(hdf_data['train_valid'], valid_index, n_classes, batch_size = batch_size, categories = cats)
            
            train_gen = gs.GeneratorSequence(np_train_valid_dataset, train_index, n_classes, batch_size = batch_size, categories = cats)
            valid_gen = gs.GeneratorSequence(np_train_valid_dataset, valid_index, n_classes, batch_size = batch_size, categories = cats)
         
            model.fit(train_gen, validation_data = valid_gen, \
                     callbacks = [slm.CheckPointer(model_name), \
                                  CSVLogger(log_path, append=True, separator=',')])
        
        
# %%
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
             [110, 190, 160],   # Static        15 -> 1
             [170, 120, 50],    # Dinamic       16 -> 1
             [45, 60, 150],     # Water         17 -> 15
             [145, 170, 100]    # Terrain       18 -> 16
             ]

test_x = hdf_data['test'][teststart:teststart+testsize,:,:,0:4]
preds = model.predict(test_x)


# for i in range(testsize):
#   pred = preds[i]
#   predcolor = np.zeros([pred.shape[0],pred.shape[1],3], dtype=np.uint8)
#   for cindex, col in enumerate(pred):
#     for lindex, labl in enumerate(col):
#         predcolor[cindex,lindex,0] = labellist[np.argmax(labl)][0]
#         predcolor[cindex,lindex,1] = labellist[np.argmax(labl)][1]
#         predcolor[cindex,lindex,2] = labellist[np.argmax(labl)][2]
#   plt.figure()
#   plt.title(str(i))
#   plt.imshow(predcolor)


# #%%
# for i in range(19):
#     plt.figure()
#     plt.imshow(preds[0,:,:,i])
#     plt.title(str(i))
    
for i in range(testsize):
    plt.figure()
    plt.suptitle(i)
    plt.subplot(preds.shape[-1]+1,1,1)
    plt.imshow(test_x[i,:,:,0:3])
    for j in range(preds.shape[-1]):
        plt.subplot(preds.shape[-1]+1, 1, j+2)
        plt.imshow(preds[i,:,:,j])


#%%
# for i in range(hdf_data['train_valid'].shape[0]):
#     plt.figure()
#     plt.imshow(hdf_data['train_valid'][i,:,:,0:3])
#     plt.title(str(i))