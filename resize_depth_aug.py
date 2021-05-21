import cv2
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import h5py
import data_augmentation as da

import glob
import re


def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)


h = 128
w = 256

carlafolder = '..\\carla_images\\'
segfolder = carlafolder + 'seg\\'
colfolder = carlafolder + 'rgb\\'
depthfolder = carlafolder + 'depth\\'

train_validf = ['001', '002', '003', '004', '005', '007']
tvfilecount = 939*6
testf = ['000', '006']
tfilecount = 298*6


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
ylist = list(range(len(labellist)))

hdf = h5py.File('..\\data\\HDF_' + str(h) + 'x' + str(w) + '_aug.h5', 'w')
hdf.create_dataset('train_valid', (tvfilecount, h, w, 5), np.uint8)
hdf.create_dataset('test', (tfilecount, h, w, 5), np.uint8)

for (column, farray) in [('train_valid', train_validf), ('test', testf)]:
    index = 0
    for findex, foname in enumerate(farray):
        print(foname)
        sfoldername = segfolder + foname + '\\'
        cfoldername = colfolder + foname + '\\'
        dfoldername = depthfolder + foname + '\\'
        names = sorted_nicely(glob.glob1(cfoldername, "*.jpg"))
        fcount = len(names)
        for fiindex, filename in enumerate(names):
            print(fiindex)
            segpath = sfoldername + filename.replace('jpg', 'png')
            colpath = cfoldername + filename
            depthpath = dfoldername + filename.replace('jpg', 'png')
          
            cimg = cv2.imread(colpath)
            dimg = cv2.imread(depthpath, flags=cv2.IMREAD_GRAYSCALE)
            simg = cv2.imread(segpath)
            cimg = cv2.cvtColor(cimg, cv2.COLOR_BGR2RGB)
            simg = cv2.cvtColor(simg, cv2.COLOR_BGR2RGB)
          
            cimg = np.array(cimg).reshape([1, 512, 1024, 3])
            dimg = np.array(dimg).reshape([1, 512, 1024, 1])
            simg = np.array(simg).reshape([1, 512, 1024, 3])
            
            # fig = plt.figure()
            # plt.subplot(311)
            # plt.imshow(cimg[0])
            # plt.subplot(312)
            # plt.imshow(dimg[0], cmap='gray', vmin=0, vmax=255)
            # plt.subplot(313)
            # plt.imshow(simg[0])
          
            if fiindex == 0:
                imgs = np.concatenate((cimg, dimg, simg), axis=3)
            else:
                imgs = np.concatenate(
                    (imgs, np.concatenate((cimg, dimg, simg), axis=3)))
        
        # plusrotate = np.zeros([findex, h, w, 5], dtype=np.uint8)
        original = da.aug_resize(imgs, w, h)
        hdf[column][index:index+fcount, ...] = original
        index += fcount
        plusrotate = da.aug_resize(da.aug_rotate(imgs, 5), w, h)
        hdf[column][index:index+fcount, ...] = plusrotate
        index += fcount
        minusrotate = da.aug_resize(da.aug_rotate(imgs, -5), w, h)
        hdf[column][index:index+fcount, ...] = minusrotate
        index += fcount
        zoom = da.aug_resize(da.aug_zoom(imgs, 1.2, 1.2), w, h)
        hdf[column][index:index+fcount, ...] = zoom
        index += fcount
        zoomplusrotate = da.aug_resize(da.aug_rotate(da.aug_zoom(imgs, 1.2, 1.2), 5), w, h)
        hdf[column][index:index+fcount, ...] = zoomplusrotate
        index += fcount
        zoomminusrotate = da.aug_resize(da.aug_rotate(da.aug_zoom(imgs, 1.2, 1.2), -5), w, h)
        hdf[column][index:index+fcount, ...] = zoomminusrotate
        index += fcount

hdf.close()
