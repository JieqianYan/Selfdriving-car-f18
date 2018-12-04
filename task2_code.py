f = open("result_new.txt", "r")

lines = f.readlines()
test = lines[0]

from glob import glob
from mpl_toolkits.mplot3d import Axes3D
import matplotlib.pyplot as plt
import numpy as np


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


f = open('TASK2.txt','w')
f.write('guid/image/axis,value\n')

a = 0
count = 0
for i in lines:
    need = i.split(",")
    files = glob(need[0])
    idx = np.random.randint(0, len(files))
    snapshot = files[idx]
    center_x = need[2]
    center_y = need[3]
    width = need[4]
    height = need[5]

    xrange = [float(center_x) - float(height)/2,float(center_x) + float(height)/2]
    yrange = [float(center_y) - float(width)/2, float(center_y) +float(width)/2]

    img = plt.imread(snapshot)
    xyz = np.fromfile(snapshot.replace('_image.jpg', '_cloud.bin'), dtype=np.float32)
    xyz = xyz.reshape([3, -1])
    proj = np.fromfile(snapshot.replace('_image.jpg', '_proj.bin'), dtype=np.float32)
    proj.resize([3, 4])

    uv = proj @ np.vstack([xyz, np.ones_like(xyz[0, :])])
    uv = uv / uv[2, :]

    needx = []
    needy = []
    needz = []

    for ii in range(len(uv[0])):
        if uv[0][ii]>=xrange[0]:
            if uv[0][ii]<=xrange[1]:
                if uv[1][ii]>=yrange[0]:
                    if uv[1][ii]<=yrange[1]:
                        needy.append(xyz[1][ii])
                        needx.append(xyz[0][ii])
                        needz.append(xyz[2][ii])
    try:
        x = ((min(needx)+max(needx))/2)
        y = ((min(needy) + max(needy))/2)
        z = ((min(needz)+max(needz))/2)
    except:
        x = 9
        y = 9
        z = 9
        count+=1

    f.write(need[0][12:-10] + "/x," + str(x)+"\n")
    f.write(need[0][12:-10] + "/y," + str(y) + "\n")
    f.write(need[0][12:-10] + "/z," + str(z) + "\n")
    a = a + 1
    print(a)
    #print(x,"\n",y,"\n",z)

f.close()
print("count=",count)








