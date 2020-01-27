# Steger algorithm for edge/line extraction
# Author : Munch Quentin, 2020

# General and computer vision lib
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot

# convert to grayscle
img = cv2.imread("desk.png")
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

# calculate hessian matrix (dx, dy, dxy) fot the current image
# first order derivative using sobel
dx = cv2.Sobel(gray_img, cv2.CV_32F, 2, 0, ksize=3)
dy = cv2.Sobel(gray_img, cv2.CV_32F, 0, 2, ksize=3)
dxy = cv2.Sobel(gray_img, cv2.CV_32F, 1, 1, ksize=3)

plt.imshow(dx)
plt.show()
plt.imshow(dy)
plt.show()
plt.imshow(dxy)
plt.show()

# create gaussian kernel
gauss_size = 10
gauss_kernel = cv2.getGaussianKernel(gauss_size, -1, cv2.CV_32F)
gauss_kernel_T = np.transpose(gauss_kernel)
# second order derivative with applied gaussian filter
ddx = cv2.sepFilter2D(dx,-1, gauss_kernel_T, gauss_kernel)
ddy = cv2.sepFilter2D(dy,-1, gauss_kernel_T, gauss_kernel)
ddxy = cv2.sepFilter2D(dxy,-1, gauss_kernel_T, gauss_kernel)

plt.imshow(ddx)
plt.show()
plt.imshow(ddy)
plt.show()
plt.imshow(ddxy)
plt.show()

# extract edge
edge = np.sqrt(dx**2 + dy**2)
edge_ = np.sqrt(ddx**2 + ddy**2)

plt.imshow(edge)
plt.show()
plt.imshow(edge_)
plt.show()

# calculate eigen vector
tmp = np.sqrt((ddx - ddy)**2 + 4*ddxy**2)
v2x = 2*ddxy
v2y = ddy - ddx + tmp
# normalization
magnitude = cv2.magnitude(ddx, ddy)
i = (magnitude != 0);
v2x[i] = v2x[i]/magnitude[i];
v2y[i] = v2y[i]/magnitude[i];
# The eigenvectors are orthogonal
v1x = -v2y
v1y = v2x
# Compute the eigenvalues
mu1 = 0.5*(ddx + ddy + tmp)
mu2 = 0.5*(ddx + ddy - tmp)
# Sort eigen values by absolute value abs(Lambda1)<abs(Lambda2)
check = np.abs(mu1) > np.abs(mu2)
Lambda1 = mu1
Lambda1[check] = mu2[check]
Lambda2 = mu2
Lambda2[check] = mu1[check]
Ix = v1x
Ix[check] = v2x[check]
Iy = v1y
Iy[check] = v2y[check]

plt.imshow(Lambda1)
plt.show()
plt.imshow(Lambda2)
plt.show()
plt.imshow(Ix)
plt.show()
plt.imshow(Iy)
plt.show()

# calculate taylor polynomial expension
T = -np.divide((dx*Ix + dy*Iy),((ddx*Ix**2)+2*ddxy*Ix*Iy+(ddy*Iy**2)), where=((ddx*Ix**2)+2*ddxy*Ix*Iy+(ddy*Iy**2))!=0)

plt.imshow(T)
plt.show()

# extract line (Px,Py)
px = T*Ix
py = T*Iy
# generate candidate point (x,y)
candidate_point = np.where((px >= -0.5) & (px <= 0.5) & (py >= -0.5) & (py <= 0.5))
PX = candidate_point[0]
PY = candidate_point[1]
