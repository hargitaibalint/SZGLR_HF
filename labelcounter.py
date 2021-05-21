import cv2
import matplotlib.pyplot as plt
import numpy as np
from os import listdir
from os.path import isfile, join
import h5py
import glob
import re

# n_classes = 17
# data = h5py.File('..\\data\\HDF_128x256_aug.h5', 'r')

# train_valid = data['train_valid']
# test = data['test']

# train_label_counts = np.zeros(n_classes, dtype = np.int)
# train_label_counts_bypic = np.zeros([len(train_valid), n_classes+1], dtype = np.int)
# test_label_counts = np.zeros(n_classes, dtype = np.int)
# test_label_counts_bypic = np.zeros([len(test), n_classes+1], dtype = np.int)

# all_label_counts = np.zeros(n_classes)

# for pindex, pix in enumerate(train_valid):
#     seq = pix[:,:,4]
#     for i in range(n_classes):
#         count = np.count_nonzero(seq == i)
#         train_label_counts_bypic[pindex, i] = count
#         train_label_counts[i] += count
        
# train_label_counts_bypic[:,17] = np.arange(len(train_valid))

# for pindex, pix in enumerate(test):
#     seq = pix[:,:,4]
#     for i in range(n_classes):
#         count = np.count_nonzero(seq == i)
#         test_label_counts_bypic[pindex, i] = count
#         test_label_counts[i] += count
        
# test_label_counts_bypic[:,17] = np.arange(len(test))

# for i in range(n_classes):
#     all_label_counts[i] += train_label_counts[i] + test_label_counts[i]

# misc = h5py.File('..\\data\\misc.h5', 'w')
# misc.create_dataset('pedestrian_indices', [1000], np.int)
# pedsort = train_label_counts_bypic[train_label_counts_bypic[:, 2].argsort()]
# pedindices = pedsort[4634:, 17]
# misc['pedestrian_indices'][...] = pedindices
# misc.close()

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

folders = ['000', '001', '002', '003', '004', '005', '006', '007']
carlafolder = '..\\carla_images\\'
segfolder = carlafolder + 'seg\\'
depthfolder = carlafolder + 'depth\\'

labellist = [[70, 70, 70],      # Building      0
             [100, 40, 40],     # Fence         1
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
             [110, 190, 160],   # Static        15
             [170, 120, 50],    # Dinamic       16
             [45, 60, 150],     # Water         17
             [145, 170, 100]    # Terrain       18
             ]

labels_hu = ['Épület', 'Kerítés', 'Járókelő', 'Villanyoszlop', 'Felfestés', 'Út', 'Járda', 'Növényzet', 'Jármű',
             'Fal', 'Közúti tábla', 'Égbolt', 'Talaj', 'Sín', 'Közúti lámpa', 'Statikus', 'Dinamikus', 'Víz', 'Terep']

labelcounts = np.zeros(len(labellist))

for foldername in folders:
    print(foldername)
    sfoldername = segfolder + foldername + '\\'
    dfoldername = depthfolder + foldername + '\\'
    # Ignore fs files
    names = sorted_nicely(glob.glob1(dfoldername, "*.png"))
    folder_pxas = []
    folder_ious = []
    folder_times = []
    for filename in names:
        segpath = sfoldername + filename
        simg = cv2.imread(segpath)
        simg = cv2.cvtColor(simg,cv2.COLOR_BGR2RGB)
        for col in simg:
            for labl in col:
                labelcounts[labellist.index(labl.tolist())] += 1

#%% - Plot counts
sortedlabelcounts = np.sort(labelcounts)[::-1]
sortedlabellist = []
for lcount in sortedlabelcounts:
    idx, = np.where(labelcounts == lcount)
    sortedlabellist.append(labels_hu[idx[0]])
    
fig = plt.figure(figsize=(12,6))
ax = fig.add_axes([0,0,1,1])
ax.grid(zorder=0)
rects = ax.bar(sortedlabellist,sortedlabelcounts)
# ax.bar_label(sortedlabelcounts, padding=3)
ax.tick_params(labelsize=16)
ax.set_xticklabels(sortedlabellist, fontsize=18, rotation=90)
ax.set_title('A különböző kategóriák előfordulási gyakorisága', fontsize=20)



plt.show()
