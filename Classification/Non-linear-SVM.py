import matplotlib.pyplot as plt
import matplotlib.Axes3D
from matplotlib import style
import numpy as np
import math

style.use('ggplot')

def transformation(x1, x2):
    return x1**2, 2*x2**2, x2

c1 = np.array([[1,1],
               [1,1],
               [1,1],
               [1,1],
               [1,1],
               [1,1],
               [1,1]])

c2 = np.array([[2,5],
             [1.5,4]])

c1_3d = np.array([transformation(x[0],x[1]) for x in c1])
c2_3d = np.array([transformation(x[0],x[1]) for x in c2])

fig = plt.figure()
ax1 = fig.add_subplot(121)
ax2 = fig.add_subplot(122, projection='3d')

ax1.scatter(c1[:,0], c1[:,1], color='r', marker='*', s=200)
ax1.scatter(c2[:,0], c2[:,1], color='b', marker='*', s=200)
ax2.scatter(c1_3d[:,0], c1_3d[:,1], c1_3d[:,2], c='r', marker='*', s=200)
ax2.scatter(c2_3d[:,0], c2_3d[:,1], c2_3d[:,2], c='b', marker='*', s=200)

plt,show()
