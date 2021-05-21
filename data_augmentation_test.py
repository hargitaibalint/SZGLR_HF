import cv2
import matplotlib.pyplot as plt
import numpy as np
from scipy import ndimage
from math import cos, sin, tan, radians
import time
import data_augmentation as da



#%% fájlok beolvasása

batch = 1
for i in range(batch):
    cimg = cv2.imread('..\\carla_images\\rgb\\002\\%02d.jpg' % i)
    dimg = cv2.imread('..\\carla_images\\depth\\002\\%02d.png' % i, flags = cv2.IMREAD_GRAYSCALE)
    simg = cv2.imread('..\\carla_images\\seg\\002\\%02d.png' % i)
    cimg = cv2.cvtColor(cimg,cv2.COLOR_BGR2RGB)
    simg = cv2.cvtColor(simg,cv2.COLOR_BGR2RGB)
    
    cimg = np.array(cimg).reshape([1, 512, 1024, 3])
    dimg = np.array(dimg).reshape([1, 512, 1024, 1])
    simg = np.array(simg).reshape([1, 512, 1024, 3])
    
    if i == 0:
        img = np.concatenate((cimg, dimg, simg), axis = 3)
    else:
        img = np.concatenate((img, np.concatenate((cimg, dimg, simg), axis = 3)))
        
#%% test

flipped = da.aug_flip(img)
brighter = da.aug_brightness(img, 100)
darker = da.aug_brightness(img, -100)
# noisy = da.aug_gauss_noise(img, 100)
rect1 = da.aug_rect(img, 0.35, 0.35, 0.5, 0.5)
rect2 = da.aug_rect(img, 0.35, 0.35, 0.95, 0.95)
rect3 = da.aug_rect(img, 0.35, 0.35, 0.05, 0.95)
rect4 = da.aug_rect(img, 0.35, 0.35, 0.05, 0.05)
rect5 = da.aug_rect(img, 0.35, 0.35, 0.95, 0.05)

zoom1 = da.aug_zoom(img, 1.2, 1.2)
zoom2 = da.aug_zoom(img, 3, 1)

rot1 = da.aug_rotate(img, -5)
rot2 = da.aug_rotate(img, 5)

resized1 = da.aug_resize(img, 256, 128)
resized2 = da.aug_resize(zoom1, 256, 128)
resized3 = da.aug_resize(rot1, 256, 128)

for (data, title) in [(img, 'Eredeti'), (flipped, 'flipped'), (brighter, 'brighter'), (darker, 'darker'),\
                      (rect1, 'rect middle'), (rect2, 'rect low right'), (rect3, 'rect low left'),\
                      (rect4, 'rect up left'), (rect5, 'rect up right'), (zoom1, 'Belenagyítva 1.2x'), (zoom2, 'zoom 3'),\
                      (rot1, 'rot -10'), (rot2, 'Elforgatva +5 fokkal'), (resized1, 'resized original'), (resized2, 'resized zoom'),\
                      (resized3, 'resized rot')]:
    fig = plt.figure(figsize=(9,4.2))
    plt.axis('off')
    fig.suptitle(title, fontsize=20, fontweight="bold")
    plt.imshow(data[0,:,:,0:3])
    
    fig = plt.figure()
    fig.suptitle(title)
    plt.subplot(311)
    plt.imshow(data[0,:,:,0:3])
    plt.subplot(312)
    plt.imshow(data[0,:,:,3:4], cmap='gray', vmin=0, vmax=255)
    plt.subplot(313)
    plt.imshow(data[0,:,:,4:7])
    
#%% execution time

count = 1000
start_time = time.time()
for i in range(count):
    dummy = da.aug_flip(resized1)
end_time = time.time()
print('Resized flip: %.3f us/batch of %d' % ((end_time-start_time)*1000000/count, batch))

count = 1
start_time = time.time()
for i in range(count):
    dummy = da.aug_brightness(resized1, np.random.randint(1,125))
end_time = time.time()
print('Resized  brightness: %.3f ms/batch of %d' % ((end_time-start_time)*1000/count, batch))

count = 1
start_time = time.time()
for i in range(count):
    dummy = da.aug_gauss_noise(resized1, np.random.randint(1,125))
end_time = time.time()
print('Resized gauss noise: %.3f ms/batch of %d' % ((end_time-start_time)*1000/count, batch))


count = 1
start_time = time.time()
for i in range(count):
    dummy = da.aug_rect(resized1, np.random.rand(), np.random.rand(), np.random.rand(), np.random.rand())
end_time = time.time()
print('Resized rect: %.3f ms/batch of %d' % ((end_time-start_time)*1000/count, batch))

count = 1
start_time = time.time()
for i in range(count):
    dummy = da.aug_zoom(img, 1 + np.random.rand(), 1 + np.random.rand())
end_time = time.time()
print('Original Zoom: %.3f ms/batch of %d' % ((end_time-start_time)*1000/count, batch))

count = 1
start_time = time.time()
for i in range(count):
    dummy = da.aug_rotate(img, 30 * np.random.rand() -15)
end_time = time.time()
print('Original Rotate: %.3f ms/batch of %d' % ((end_time-start_time)*1000/count, batch))

count = 1
start_time = time.time()
for i in range(count):
    dummy = da.aug_resize(img, 258, 128)
end_time = time.time()
print('Original Resize: %.3f ms/batch of %d' % ((end_time-start_time)*1000/count, batch))
    
#%% resize test

resized_none = cv2.resize(img[0,:,:,0:3], (128, 64), 0, 0)
resized_area = cv2.resize(img[0,:,:,0:3], (128, 64), 0, 0, interpolation = cv2.INTER_AREA)
resized_lanczos = cv2.resize(img[0,:,:,0:3], (128, 64), 0, 0, interpolation = cv2.INTER_LANCZOS4)

plt.figure()
plt.imshow(img[0,:,:,0:3])
plt.title('original')


plt.figure()
plt.imshow(resized_none)
plt.title('interpolation none')

plt.figure()
plt.imshow(resized_area)
plt.title('interpolation area')

plt.figure()
plt.imshow(resized_lanczos)
plt.title('interpolation lanczos')