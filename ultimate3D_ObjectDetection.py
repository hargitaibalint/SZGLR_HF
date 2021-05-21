import cv2
import matplotlib.pyplot as plt
import numpy as np
import copy
import glob
import re

carlafolder = '..\\carla_images\\'
segfolder = carlafolder + 'seg\\'
colfolder = carlafolder + 'rgb\\'
depthfolder = carlafolder + 'depth\\'
annotationfolder = carlafolder + 'annotation\\'


annotation2label = [[0, 0, 142],[0, 0, 142],[0, 0, 142],[0, 0, 142],[220, 20, 60]]
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

# bb_max = np.zeros(bb0.shape[1])
# for i in range(bb0.shape[0]):
#     if bb0[i,3]*bb0[i,4] > bb_max[3]*bb_max[4]:
#         bb_max = bb0[i]

# rect_x = bb_max[1]*img.shape[1]
# rect_y = bb_max[2]*img.shape[0]
# rect_w = bb_max[3]*img.shape[1]
# rect_h = bb_max[4]*img.shape[0]

# p1 = (int(rect_x-rect_w/2), int(rect_y-rect_h/2))
# p2 = (int(rect_x+rect_w/2), int(rect_y+rect_h/2))

# car_rgb = img[p1[1]:p2[1], p1[0]:p2[0],:]
# car_depth = depth[p1[1]:p2[1], p1[0]:p2[0]]
# fig = plt.figure()
# plt.subplot(211)
# plt.imshow(car_rgb)
# plt.subplot(212)
# plt.imshow(car_depth, cmap='gray')

cam_mtx = np.array([[358.5, 0.0,   512.0],
                    [0.0,   358.5, 256.0],
                    [0.0,   0.0,   1.0]])

none = ['none']

def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)

def xy_from_depth(depth, cam_matrix = cam_mtx):
        """
        Computes the x, and y coordinates of every pixel in the image using the depth map and the calibration matrix.
        """
        
        
        ### START CODE HERE ### (≈ 7 lines in total)
        # Get the shape of the depth tensor
        # print(depth.shape)
        H, W = np.shape(depth)
        # Grab required parameters from the cam matrix
        f = cam_matrix[0,0]
        c_u = cam_matrix[0,2]
        c_v = cam_matrix[1,2]
        # Generate a grid of coordinates corresponding to the shape of the depth map
        x = np.zeros((H, W))
        y = np.zeros((H, W))
        # Compute x and y coordinates
        for i in range(H):
            for j in range(W):
                d = depth[i, j]
                x[i, j] = ((j+1 - c_u)*d) / f
                y[i, j] = ((i+1 - c_v)*d) / f
        return x, y
    
def point3Dto2D(point, cam_matrix = cam_mtx):
    fx = cam_matrix[0,0]
    fy = cam_matrix[1,1]
    px = cam_matrix[0,2]
    py = cam_matrix[1,2]
    
    return (int(fx*point[0]/point[2] + px), int(fy*(point[1]/point[2]) + py))
    
    
def mask_squeeze(mask, radius = 1):
    ret = copy.deepcopy(mask)
    inverse = ~mask
    for i in range(ret.shape[0]):
        for j in range(ret.shape[1]):
            x1 = max(0, j-radius)
            x2 = min(ret.shape[1], j + radius +1)
            y1 = max(0, i - radius)
            y2 = min(ret.shape[0], i + radius +1)
            if ret[i,j] and np.sum(inverse[y1:y2,x1:x2]) != 0:
                ret[i,j] = False
    return ret

# box: [x1, x2, y1, y2, z1, z2]
def iou3D(box1, box2):
    if box1 == none or box2 == none:
        return -1
    
    for i in range(3):
        box1[2*i:2*(i+1)] = sorted(box1[2*i:2*(i+1)])
        box2[2*i:2*(i+1)] = sorted(box2[2*i:2*(i+1)])
    dxi = min(box1[1], box2[1]) - max(box1[0], box2[0])
    dyi = min(box1[3], box2[3]) - max(box1[2], box2[2])
    dzi = min(box1[5], box2[5]) - max(box1[4], box2[4])
    if dxi < 0 or dyi < 0 or dzi < 0:
        vi = 0
    else:
        vi = dxi*dyi*dzi # volume of the intersection
    # print('volume of the intersection: %.4f' % (vi))
    dx1 = box1[1] - box1[0]
    dy1 = box1[3] - box1[2]
    dz1 = box1[5] - box1[4]
    dx2 = box2[1] - box2[0]
    dy2 = box2[3] - box2[2]
    dz2 = box2[5] - box2[4]
    vu = dx1*dy1*dz1 + dx2*dy2*dz2 - vi # volume of the union
    # print('volume of the union: %.4f' % (vu))
    return vi/vu

# 3D-s bounding box számítás a maskolt objektumra a mélység kép alapján
# return box: [x1, x2, y1, y2, z1, z2]
def calc_BB3D(X_coords, Y_coords, Z_coords, mask):    
    box = []
    # TODO: szélsőértékek kizárása mask_squeeze helyett
    xs = [pix for i,row in enumerate(X_coords) for j,pix in enumerate(row) if mask[i, j]]
    ys = [pix for i,row in enumerate(Y_coords) for j,pix in enumerate(row) if mask[i, j]]
    zs = [pix for i,row in enumerate(Z_coords) for j,pix in enumerate(row) if mask[i, j]]
    
    # fig = plt.figure(figsize=(12,16))
    # plt.subplot(211)
    # plt.title('Szűrés előtt', fontsize=36)
    # plt.plot(xs)
    # plt.plot(ys)
    # plt.plot(zs)
    # plt.tick_params(axis='both', which='major', labelsize=30)
    # plt.tick_params(axis='both', which='minor', labelsize=26)
    # # plt.xlabel('Pixel', fontsize=30)
    # plt.ylabel('Távolság [m]', fontsize=30)
    # plt.legend(['X', 'Y', 'Z'], loc='upper right', fontsize=30)
    
    x_c = X_coords[X_coords.shape[0]//2, X_coords.shape[1]//2]
    y_c = Y_coords[Y_coords.shape[0]//2, Y_coords.shape[1]//2]
    z_c = Z_coords[Z_coords.shape[0]//2, Z_coords.shape[1]//2]
    
    # print('center point: %.4f %.4f %.4f' % (x_c, y_c, z_c))
    
    xs_fil = [xs[i] for  i in range(len(xs)) if ((xs[i]-x_c)**2 + (ys[i]-y_c)**2 + (zs[i]-z_c)**2) < 25]
    ys_fil = [ys[i] for  i in range(len(ys)) if ((xs[i]-x_c)**2 + (ys[i]-y_c)**2 + (zs[i]-z_c)**2) < 25]
    zs_fil = [zs[i] for  i in range(len(zs)) if ((xs[i]-x_c)**2 + (ys[i]-y_c)**2 + (zs[i]-z_c)**2) < 25]
    
    
    # plt.subplot(212)
    # plt.title('Szűrés után', fontsize=36)
    # plt.plot(xs_fil)
    # plt.plot(ys_fil)
    # plt.plot(zs_fil)
    # plt.tick_params(axis='both', which='major', labelsize=30)
    # plt.tick_params(axis='both', which='minor', labelsize=26)
    # plt.xlabel('Pixel', fontsize=30)
    # plt.ylabel('Távolság [m]', fontsize=30)
    # plt.legend(['X', 'Y', 'Z'], loc='center right', fontsize=30)
    
    # plt.figure()
    # plt.title('utána')
    # plt.plot(xs_fil)
    # plt.plot(ys_fil)
    # plt.plot(zs_fil)
    if len(xs_fil) == 0 or len(ys_fil) == 0 or len(zs_fil) == 0:
        return none
    box.append(np.min(xs_fil))
    box.append(np.max(xs_fil))
    box.append(np.min(ys_fil))
    box.append(np.max(ys_fil))
    box.append(np.min(zs_fil))
    box.append(np.max(zs_fil))
    
    return box

# box: [x1, x2, y1, y2, z1, z2]
def draw_BB3D(img, box):
    if box == none:
        return
    
    for i in range(3):
        box[2*i:2*(i+1)] = sorted(box[2*i:2*(i+1)])

    p_3Ds = np.array([[box[0], box[2], box[4]], [box[1], box[2], box[4]], [box[0], box[3], box[4]], [box[1], box[3], box[4]],
             [box[0], box[2], box[5]], [box[1], box[2], box[5]], [box[0], box[3], box[5]], [box[1], box[3], box[5]]])
    
    p_2Ds = []
    for p in p_3Ds:
        p_2Ds.append(point3Dto2D(p))
        
    # d2_s = [p[0]**2 + p[1]**2 + p[2]**2 for p in p_3Ds[:4]]
    
    for i in range(p_3Ds.shape[0]):
        for j in range(i, p_3Ds.shape[0]):
            if np.sum(p_3Ds[i] == p_3Ds[j]) == 2:
                cv2.line(img, p_2Ds[i], p_2Ds[j], (255,0,0), 2)
        
    

folder = '000\\'
files = [f[:-4] for f in sorted_nicely(glob.glob1(colfolder + folder, "*.jpg"))]

# files = files[:1]
all_BBs = []
for f in files:

    # fájlok beolvasása
    img = cv2.imread(colfolder + folder + f +'.jpg')
    img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
    seg = cv2.imread(segfolder + folder + f +'.png')
    seg = cv2.cvtColor(seg, cv2.COLOR_BGR2RGB)
    depth = cv2.imread(depthfolder + folder + f +'.png', cv2.IMREAD_UNCHANGED)
    pos = np.load(annotationfolder + folder + f +'_.npy').reshape((4,4)) # TODO: ez mi a faszra jó?
    bb_file = open(annotationfolder + folder + f +'_.txt', 'r')
    bb_lines = bb_file.readlines()
    bb_file.close()
    
    depth = depth / 100.00 # méterben!!
    img_x, img_y = xy_from_depth(depth)
    
    bbs = []
    for l in bb_lines:
        bbs.append(l.split(' ')[:-1])
    bbs = np.array(bbs, dtype=np.float32)
    
    
    for i,bb in enumerate(bbs):
        rect_x = bb[1]*img.shape[1]
        rect_y = bb[2]*img.shape[0]
        rect_w = bb[3]*img.shape[1]
        rect_h = bb[4]*img.shape[0]
        
        p1 = np.array([rect_x-rect_w/2, rect_y-rect_h/2], dtype = np.int)
        p2 = np.array([rect_x+rect_w/2, rect_y+rect_h/2], dtype = np.int)
        
        p1[0] = np.clip(p1[0], 0, img.shape[1])
        p1[1] = np.clip(p1[1], 0, img.shape[0])
        p2[0] = np.clip(p2[0], 0, img.shape[1])
        p2[1] = np.clip(p2[1], 0, img.shape[0])
        
        
        obj_rgb = img[p1[1]:p2[1], p1[0]:p2[0],:]    
        obj_mask = (np.sum((seg[:,:] == annotation2label[int(bb[0])]), axis = 2) == 3)[p1[1]:p2[1], p1[0]:p2[0]]
        obj_depth = np.log(depth[p1[1]:p2[1], p1[0]:p2[0]])
        
        # fig = plt.figure(figsize=(10,8))
        # plt.subplot(211)
        # plt.title('Mélységkép', fontsize=24)
        # plt.axis('off')
        # plt.imshow(obj_depth, cmap='gray')
        
        # plt.subplot(212)
        # plt.title('Maszk', fontsize=24)
        # plt.axis('off')
        # plt.imshow(obj_mask, cmap='gray')
        
        my_BB3D = []
        my_BB3D.append(folder + f)
        my_BB3D.append(bb[0])
        my_BB3D.append(calc_BB3D(img_x[p1[1]:p2[1], p1[0]:p2[0]], img_y[p1[1]:p2[1], p1[0]:p2[0]], depth[p1[1]:p2[1], p1[0]:p2[0]], obj_mask))
        my_BB3D.append(iou3D(bb[5:11], my_BB3D[2]))    # calculate the iou metric
        # print('iou for %d. obj is: %.4f' % (i+1, my_BB3D[3]))
        
        all_BBs.append(my_BB3D)
        
    for bb in all_BBs:
        if bb[0] == (folder+f):
            draw_BB3D(img, bb[2])
            

    # fig = plt.figure(figsize=(9,4.2))
    # plt.axis('off')
    # # plt.title(folder+f)
    # plt.imshow(img)
        
    

ious = []
for kaki in all_BBs:
    print(kaki[3])
    ious.append(kaki[3])
    
ious.sort()
ious = [iou for iou in ious  if iou > -0.01]
plt.figure(figsize=(10,8))
plt.tick_params(axis='both', which='major', labelsize=30)
plt.tick_params(axis='both', which='minor', labelsize=26)
plt.grid()
plt.xlim([0, len(ious)])
plt.ylim([0,1])
plt.title('IoU eloszlása nappali teszten', fontsize=30)
plt.plot(ious, linewidth=5)


#%% d_test
# TODO? neurális hálónap floatra konvertálva logaritmikus skálán a depth
'''
d_test = depth + 65535*(depth == 0) # ég kicserelésére maximumra
d_test = np.log(d_test)/np.log(2**(1/16)) # átskálázás 0-256 közé
d_test = d_test/256 # átskálázás 0-1 közé
plt.imshow(d_test, cmap='gray')
'''
