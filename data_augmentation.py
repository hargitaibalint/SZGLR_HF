import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from math import cos, sin, tan, radians
import time

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

#%% online augmentáció
# len(data.shape) == 4 van feltételezve
# data.dtype == np.uint8 van feltételezve
# [:,:,:,0:3] - RGB, [:,:,:,3:4] - depth, [:,:,:,4:7] - segmentation
def aug_flip(data):
    ret = np.flip(data, axis=2)
    return ret

# def aug_brightness(data, value):
#     cimg = data[:,:,:,0:3]
#     if value >= 0:
#         cimg = np.clip(cimg, 0, 255 - value, dtype = np.uint8)
#     else:
#         cimg = np.clip(cimg, -value, 255, dtype = np.uint8)
#     cimg = cimg + value*np.ones(cimg.shape, dtype = np.uint8)
#     ret = np.concatenate((cimg, data[:,:,:,3:]), axis = 3)
#     return ret

def aug_brightness(data, value):
    cimg = np.array(data[:,:,:,0:3], dtype=np.int)
                  
    cimg = cimg + value*np.ones(cimg.shape, dtype = np.int)
         
    cimg = np.clip(cimg, 0,255)
    cimg = np.array(cimg, dtype=np.uint8)
    ret = np.concatenate((cimg, data[:,:,:,3:]), axis = 3)
    return ret

def aug_gauss_noise(data, std_dev):
    cimg = np.array(data[:,:,:,0:3], dtype=np.int)
    noise = np.random.normal(0, std_dev, cimg.shape)
    noise = np.array(noise, dtype=np.int)
    cimg = cimg + noise
    cimg = np.clip(cimg, 0,255)
    cimg = np.array(cimg, dtype=np.uint8)
    ret = np.concatenate((cimg, data[:,:,:,3:]), axis = 3)
    return ret

# TODO: depthre másmilyen téglalap?
def aug_rect(data, rect_x, rect_y, pos_x, pos_y):
    low_x = pos_x*data.shape[2] - rect_x*data.shape[2]/2
    high_x = pos_x*data.shape[2] + rect_x*data.shape[2]/2
    low_y = pos_y*data.shape[1] - rect_y*data.shape[1]/2
    high_y = pos_y*data.shape[1] + rect_y*data.shape[1]/2
    
    [low_x, high_x] = np.clip([low_x, high_x], 0, data.shape[2])
    [low_y, high_y] = np.clip([low_y, high_y], 0, data.shape[1])
    [low_x, high_x, low_y, high_y] = np.round([low_x, high_x, low_y, high_y]).astype(np.int)
    
    ret = np.array(data)
    ret[:, low_y:high_y, low_x:high_x,:] = np.zeros([data.shape[0], high_y-low_y, high_x-low_x, data.shape[3]], dtype=np.uint8)
    return ret

#%% offline augmentáció
# len(data.shape) == 4 van feltételezve
# data.dtype == np.uint8 van feltételezve
# [:,:,:,0:3] - RGB, [:,:,:,3:4] - depth, [:,:,:,4:7] - segmentation
# TODO resize?
def aug_zoom(data, zoom_x, zoom_y):
    low_x = np.round((zoom_x-1)/(2*zoom_x)*data.shape[2]).astype(np.int)
    high_x = np.round((zoom_x+1)/(2*zoom_x)*data.shape[2]).astype(np.int)
    low_y = np.round((zoom_y-1)/(2*zoom_y)*data.shape[1]).astype(np.int)
    high_y = np.round((zoom_y+1)/(2*zoom_y)*data.shape[1]).astype(np.int)
    
    ret = np.array(data[:, low_y:high_y, low_x:high_x,:], dtype=np.uint8)
    return ret
    
# angle: -90...90
def aug_rotate(data, angle):
    ret = np.array(data)
    ret[:,:,:,0:3] = ndimage.rotate(ret[:,:,:,0:3], angle, axes=(1,2), reshape = False, order=3)
    ret[:,:,:,3:4] = ndimage.rotate(ret[:,:,:,3:4], angle, axes=(1,2), reshape = False, order=3)
    ret[:,:,:,4:7] = ndimage.rotate(ret[:,:,:,4:7], angle, axes=(1,2), reshape = False, order=0)
    
    # zoom értékek meghatározása
    z1 = cos(radians(angle))+abs(sin(radians(angle)))*data.shape[2]/data.shape[1]
    z2 = cos(radians(angle))+abs(sin(radians(angle)))*data.shape[1]/data.shape[2]
    zoom = max(z1,z2)
    # print('zoom: %.5f' % (zoom))
    ret = aug_zoom(ret, zoom, zoom)
    return ret

# return shape [:,height,width,0:3] - RGB, [:,height,width,3:4] - depth, [:,height,width,4:7] - segmentation
def aug_resize(data, width, height):
    ret = np.zeros((data.shape[0], height, width, 5), dtype=np.uint8)
    for i in range(data.shape[0]):
        ret[i,:,:,0:3]  = cv2.resize(data[i,:,:,0:3], (width, height), 0, 0, interpolation = cv2.INTER_LANCZOS4)
        ret[i,:,:,3]    = cv2.resize(data[i,:,:,3:4], (width, height), 0, 0, interpolation = cv2.INTER_LANCZOS4)
        simgres  = cv2.resize(data[i,:,:,4:7], (width, height), 0, 0, interpolation = cv2.INTER_NEAREST)
        yres = np.zeros((simgres.shape[0], simgres.shape[1], 1), dtype=np.uint8)
        for cindex, col in enumerate(simgres):
            for pindex, pix in enumerate(col):
              idx = labellist.index(pix.tolist())
              if idx in [15,16]:
                  idx = 1
              if idx == 17:
                  idx = 15
              if idx == 18:
                  idx = 16
              yres[cindex,pindex] = idx
        ret[i,:,:,4:5] = yres
    return ret
