import numpy as np
from os import rename
from shutil import move
import glob
import re


def sorted_nicely(l):
    """ Sort the given iterable in the way that humans expect."""
    def convert(text): return int(text) if text.isdigit() else text

    def alphanum_key(key): return [convert(c)
                                   for c in re.split('([0-9]+)', key)]
    return sorted(l, key=alphanum_key)


h = 128
w = 256

carlafolder = '..\\carla_images\\'
colfolder = carlafolder + 'rgb\\'
annfolder = carlafolder + 'annotation\\'
imgdatafolder = '..\\data\\carla\\images\\'
labeldatafolder = '..\\data\\carla\\labels\\'



train_validf = ['001', '002', '003', '004', '005', '007']
testf = ['000', '006']

# for (savefoldername, farray) in [('train\\', train_validf), ('val\\', testf)]:
#     index = 0
#     for findex, foname in enumerate(farray):
#         cfoldername = colfolder + foname + '\\'
#         names = sorted_nicely(glob.glob1(cfoldername, "*.jpg"))
#         for fname in names:
#             oldf = cfoldername + fname
#             newf = cfoldername + 'img' + str(index) + '.jpg'
#             rename(oldf, newf)
#             move(newf, imgdatafolder + savefoldername)
#             index += 1

for (savefoldername, farray) in [('train\\', train_validf), ('val\\', testf)]:
    index = 0
    for findex, foname in enumerate(farray):
        afoldername = annfolder + foname + '\\'
        names = sorted_nicely(glob.glob1(afoldername, "*.txt"))
        for fname in names:
            oldf = afoldername + fname
            rfile = open(oldf, 'r')
            rLines = rfile.readlines()
            rfile.close()
            wLines = []
            for rl in rLines:
                rlcoords = rl.split()[1:5]
                skipline = False
                for rc in rlcoords:
                    if float(rc) < 0 or float(rc) > 1:
                        skipline = True
                if skipline:
                    continue
                wl = ' '
                wl = wl.join(rl.split()[0:5])
                wLines.append(wl)
            newf = labeldatafolder + savefoldername + 'img' + str(index) + '.txt'
            wfile = open(newf, 'w')
            for wl in wLines:
                wfile.writelines(wl + '\n')
            wfile.close()
            index += 1
            