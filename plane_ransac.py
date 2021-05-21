#%%
import cv2
import matplotlib.pyplot as plt
import numpy as np
import time

import glob
import re

K = np.array([
    [358.5, 0.0,   512.0],
    [0.0,   358.5, 256.0],
    [0.0,   0.0,   1.0],
])

def xy_from_depth(depth):
        """
        Computes the x, and y coordinates of every pixel in the image using the depth map and the calibration matrix.
        """
        ### START CODE HERE ### (≈ 7 lines in total)
        # Get the shape of the depth tensor
        H, W = np.shape(depth)
        # Grab required parameters from the cam matrix
        f = K[0,0]
        p_x = K[0,2]
        p_y = K[1,2]
        # Generate a grid of coordinates corresponding to the shape of the depth map
        x = np.zeros((H, W))
        y = np.zeros((H, W))
        # Compute x and y coordinates
        for v in range(H):
            for u in range(W):
                z = depth[v, u]
                x[v, u] = ((u - p_x)*z) / f
                y[v, u] = ((v - p_y)*z) / f
                
        return x, y

def compute_plane(xyz):
    """
    Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
    Arguments:
    """  
    p0 = xyz[0]
    p1 = xyz[1]
    p2 = xyz[2]
    x0, y0, z0 = p0
    x1, y1, z1 = p1
    x2, y2, z2 = p2
    
    ux, uy, uz = [x1-x0, y1-y0, z1-z0]
    vx, vy, vz = [x2-x0, y2-y0, z2-z0]
    
    u_cross_v = [uy*vz-uz*vy, uz*vx-ux*vz, ux*vy-uy*vx]
    
    point  = np.array(p0)
    normal = np.array(u_cross_v)
    
    d = -point.dot(normal)
    
    pl2 = np.append(normal, d)

    return pl2

def dist_to_plane(plane, x, y, z):
    """
    Computes distance between points provided by their x, and y, z coordinates
    and a plane in the form ax+by+cz+d = 0
    """
    a, b, c, d = plane

    return np.abs(a * x + b * y + c * z + d) / np.sqrt(a**2 + b**2 + c**2)

def ransac_plane_fit(xyz_data, th):
    """
    Computes plane coefficients a,b,c,d of the plane in the form ax+by+cz+d = 0
    using ransac for outlier rejection.
    """
    
    # Set thresholds:
    num_itr = 10  # RANSAC maximum number of iterations
    distance_threshold = th     # Maximum distance from point to plane for point to be considered inlier
    largest_number_of_inliers = 0
    largest_inlier_set_indexes = 0

    for h in range(num_itr):
        # Step 1: Choose a minimum of 3 points from xyz_data at random.
        indexes = [np.random.choice(np.arange(200,xyz_data[0].shape[0]), 3, replace = False), np.random.choice(xyz_data[0].shape[1], 3, replace = False)]
        pt1 = np.array([xyz_data[0][indexes[0][0], indexes[1][0]], xyz_data[1][indexes[0][0], indexes[1][0]], xyz_data[2][indexes[0][0], indexes[1][0]]])
        pt2 = np.array([xyz_data[0][indexes[0][1], indexes[1][1]], xyz_data[1][indexes[0][1], indexes[1][1]], xyz_data[2][indexes[0][1], indexes[1][1]]])
        pt3 = np.array([xyz_data[0][indexes[0][2], indexes[1][2]], xyz_data[1][indexes[0][2], indexes[1][2]], xyz_data[2][indexes[0][2], indexes[1][2]]])
        pts = [pt1, pt2, pt3]
    
        # Step 2: Compute plane model
        p = compute_plane(pts)
    
        # Step 3: Find number of inliers
        distance = np.zeros((512,1024))+distance_threshold
        inlinerstoplot = []
        for i in range(200,512):
            for j in range(1024):
                d = dist_to_plane(p, xyz_data[0][i,j], xyz_data[1][i,j], xyz_data[2][i,j])
                distance[i,j] = d
                if d < distance_threshold:
                    inlinerstoplot.append([i,j])
        number_of_inliers = len(distance[distance < distance_threshold])
        # print(number_of_inliers)
    
        # Step 4: Check if the current number of inliers is greater than all previous iterations and keep the inlier set with the largest number of points.
        if number_of_inliers > largest_number_of_inliers:
            largest_number_of_inliers = number_of_inliers
            largest_inlier_set_indexes = np.where(distance < distance_threshold)[0]
            bestplane = p
            bestinlinerstoplot = inlinerstoplot

    output_plane = bestplane
    
    inliners = np.zeros((512,1024))
    for idc in bestinlinerstoplot:
        inliners[idc[0], idc[1]] = xyz_data[2][idc[0], idc[1]]
    bwimg = np.zeros((512,1024,3), dtype=np.uint8)
    for idc in bestinlinerstoplot:
        bwimg[idc[0], idc[1],:] = np.array([255,255,255], dtype=np.uint8)
    # print(len(bestinlinerstoplot))
    # fig = plt.figure()
    # plt.subplot(211)
    # plt.imshow(inliners)
    # plt.subplot(212)
    # plt.imshow(colimg)

    return output_plane, bwimg

def ransac(depth, th) :
    """
    Show sementic segmentation image
    """
        
    z = depth
    x, y = xy_from_depth(z)

    xyz_ground = [x, y, z]

    p_final, bwimg = ransac_plane_fit(xyz_ground, th)
    
    # y[0:200, :] = 10000
    
    # fig = plt.figure(figsize=(10,18))
    # fig.suptitle('Lejtős terep', fontsize = 30, y=0.92)
    # plt.subplot(311)
    # plt.title('X koordináták', fontsize=20)
    # plt.imshow(x)
    # plt.subplot(312)
    # plt.title('Y koordináták', fontsize=20)
    # plt.imshow(y)
    # plt.subplot(313)
    # plt.title('Z koordináták', fontsize=20)
    # plt.imshow(z)
    
    # fig.set_figheight(30)
    # fig.set_figwidth(10)
    
    # fig = plt.figure()
    # plt.imshow(dist)
    
    
    return bwimg

def sorted_nicely( l ):
    """ Sort the given iterable in the way that humans expect."""
    convert = lambda text: int(text) if text.isdigit() else text
    alphanum_key = lambda key: [ convert(c) for c in re.split('([0-9]+)', key) ]
    return sorted(l, key = alphanum_key)

def calc_pxa(p1, p2):
    correct_px = 0
    for i in range(p1.shape[0]):
        for j in range(p1.shape[1]):
            if p1[i,j,0] == p2[i,j,0]:
                correct_px += 1
    return correct_px / (p1.shape[0]*p1.shape[1])

def calc_iou(p1, p2):
    intersect = 0
    union = 0
    for i in range(p1.shape[0]):
        for j in range(p1.shape[1]):
            if p1[i,j,0] == 255 and p2[i,j,0] == 255:
                intersect += 1
                union += 1
            elif p1[i,j,0] != p2[i,j,0]:
                union += 1     
    return intersect / union
    
#%%


folders = ['000', '001', '002', '003', '004', '005', '006', '007']
carlafolder = '..\\carla_images\\'
segfolder = carlafolder + 'seg\\'
depthfolder = carlafolder + 'depth\\'

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
             [110, 190, 160],   # Static        15
             [170, 120, 50],    # Dinamic       16
             [45, 60, 150],     # Water         17
             [145, 170, 100],   # Terrain       18
             ]

thresultss = []
ths = [15,25,35]
for th in ths:
    print('Distance threshold: ' + str(th))
    pxas = []
    ious = []
    times = []
    for foldername in folders:
        print(foldername)
        sfoldername = segfolder + foldername + '\\'
        dfoldername = depthfolder + foldername + '\\'
        names = sorted_nicely(glob.glob1(dfoldername, "*.png"))
        folder_pxas = []
        folder_ious = []
        folder_times = []
        for filename in names[0:50]:
            stime = time.time()
            fspath = sfoldername + 'fs_' + filename
            dpath = dfoldername + filename
            
            fs = cv2.imread(fspath)
            depth = cv2.imread(dpath, -1)
            
            bw = ransac(depth, th)
            
            plt.figure(figsize=(10,4))
            plt.subplot(211)
            plt.title('Valódi')
            plt.imshow(fs)
            plt.subplot(212)
            plt.title('Becsült')
            plt.imshow(bw)
            
            folder_pxas.append(calc_pxa(fs, bw))
            folder_ious.append(calc_iou(fs, bw))
            etime = time.time()
            
            dur = etime - stime
            folder_times.append(dur)
            
        pxas.append(folder_pxas)
        ious.append(folder_ious)
        times.append(folder_times)
    thresultss.append([pxas, ious, times])    
        

#%% Plots

pxas_to_plot = [[] for _ in range(len(thresults))]
ious_to_plot = [[] for _ in range(len(thresults))]
for tindex, thres in enumerate(thresults):
    pxa = thres[0]
    for pxal in pxa:
        pxas_to_plot[tindex].append(np.mean(pxal))
    iou = thres[1]
    for ioul in iou:
        ious_to_plot[tindex].append(np.mean(ioul))

# Threshold results
xlabels = folders
plt.figure(figsize=(10,8))
plt.plot(xlabels, pxas_to_plot[0], label="pxa_15m", color = "#b3d9ff")
plt.plot(xlabels, pxas_to_plot[1], label="pxa_25m", color = "#1a8cff")
plt.plot(xlabels, pxas_to_plot[2], label="pxa_35m", color = "#004080")

plt.plot(xlabels, ious_to_plot[0], label="IoU_15m", color = "#ffd1b3")
plt.plot(xlabels, ious_to_plot[1], label="IoU_25m", color = "#ff751a")
plt.plot(xlabels, ious_to_plot[2], label="IoU_35m", color = "#803300")
plt.xticks(fontsize=16)
plt.yticks(fontsize=16)
plt.legend()



# Other plots

# fig = plt.figure(figsize=(8,12))
# plt.title("Síkillesztés sík terepen", fontsize=16)
# plt.subplot(211)
# plt.imshow(bw)
# plt.subplot(212)
# plt.imshow(gd)

# fig = plt.figure()
# plt.imshow(z)

# Experimental 3D plot
# fig = plt.figure()
# ax = fig.add_subplot(111,projection='3d')
# xx, yy = np.meshgrid(range(10), range(10))
# z = (-p[3] - p[0]*xx - p[1]*yy) / p[2]
# ax.plot_surface(xx, yy, z, alpha=0.5)
# plt.show()
