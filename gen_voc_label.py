#! /usr/bin/python3
from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import csv
import matplotlib.pyplot as plt
import numpy as np
import xml.etree.ElementTree as ET
import pickle
import os
from os import listdir, getcwd
from os.path import join

classes = (
    'Unknown', 'Compacts', 'Sedans'
)

sets=[('trainval'),('test')]
#sets=[('test')]
def rot(n):
    n = np.asarray(n).flatten()
    assert(n.size == 3)

    theta = np.linalg.norm(n)
    if theta:
        n /= theta
        K = np.array([[0, -n[2], n[1]], [n[2], 0, -n[0]], [-n[1], n[0], 0]])

        return np.identity(3) + np.sin(theta) * K + (1 - np.cos(theta)) * K @ K
    else:
        return np.identity(3)


def get_bbox(p0, p1):
    """
    Input:
    *   p0, p1
        (3)
        Corners of a bounding box represented in the body frame.

    Output:
    *   v
        (3, 8)
        Vertices of the bounding box represented in the body frame.
    *   e
        (2, 14)
        Edges of the bounding box. The first 2 edges indicate the `front` side
        of the box.
    """
    v = np.array([
        [p0[0], p0[0], p0[0], p0[0], p1[0], p1[0], p1[0], p1[0]],
        [p0[1], p0[1], p1[1], p1[1], p0[1], p0[1], p1[1], p1[1]],
        [p0[2], p1[2], p0[2], p1[2], p0[2], p1[2], p0[2], p1[2]]
    ])
    e = np.array([
        [2, 3, 0, 0, 3, 3, 0, 1, 2, 3, 4, 4, 7, 7],
        [7, 6, 1, 2, 1, 2, 4, 5, 6, 7, 5, 6, 5, 6]
    ], dtype=np.uint8)

    return v, e

def convert_annotation(snapshot, image_set):
    proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])

    try:
        bbox = np.fromfile(snapshot.replace('_image.jpg', '_bbox.bin'), dtype=np.float32)
    except FileNotFoundError:
        print('[*] bbox not found.')
        bbox = np.array([], dtype=np.float32)

    bbox = bbox.reshape([-1, 11])
    temp = snapshot
    img = temp.split("/")
    img_name = img[-1].split(".")[0]
    
    #if not os.path.exists('deploy/%s/%s/labels/'%(image_set, img[-3])):
    #    os.makedirs('deploy/%s/%s/labels/'%(image_set, img[-3]))
   
    dw = 1./1914
    dh = 1./1052
    label_file = csv.reader(open("labels.csv"), delimiter=",")
    this_class = -1
    for fname, img_class in label_file:
        file_name = ('%s/%s'%(img[-2], img_name))
        if fname == file_name:
            print("get label\n")
            this_class = img_class
            break

    out_file = open('deploy/%s/%s/%s.txt'%(image_set, img[-2], img_name), 'w')
    for k, b in enumerate(bbox):
        R = rot(b[0:3])
        t = b[3:6]

        sz = b[6:9]
        vert_3D, edges = get_bbox(-sz / 2, sz / 2)
        vert_3D = R @ vert_3D + t[:, np.newaxis]

        vert_2D = proj @ np.vstack([vert_3D, np.ones(vert_3D.shape[1])])
        vert_2D = vert_2D / vert_2D[2, :]

        b_0 = float('+inf')
        b_1 = 0
        b_2 = float('+inf')
        b_3 = 0
        for e in edges.T:
            b_0 = min(b_0, min(vert_2D[0, e[0]], vert_2D[0, e[1]]))
            b_1 = max(b_1, max(vert_2D[0, e[0]], vert_2D[0, e[1]]))
            b_2 = min(b_2, min(vert_2D[1, e[0]], vert_2D[1, e[1]]))
            b_3 = max(b_3, max(vert_2D[1, e[0]], vert_2D[1, e[1]]))

        x = (b_0 + b_1)/2.0
        y = (b_2 + b_3)/2.0
        w = b_1 - b_0
        h = b_3 - b_2
        x = x*dw
        w = w*dw
        y = y*dh
        h = h*dh
        #c = classes[int(b[9])]
	
        out_file.write(str(this_class) + " " + " ".join([str(a) for a in (x,y,w,h)]) + '\n')


wd = getcwd()

for image_set in sets:
    #if not os.path.exists('deploy/%s/labels/'%(image_set)):
    #    os.makedirs('deploy/%s/labels/'%(image_set))
    files = glob('deploy/%s/*/*_image.jpg'%(image_set))
    list_file = open('%s.txt'%(image_set), 'w')
    for idx in range(len(files)):
        list_file.write('%s/%s\n'%(wd, files[idx]))
        convert_annotation(files[idx], image_set)
    list_file.close()

