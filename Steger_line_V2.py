# Steger algorithm for edge/line extraction
# Author : Munch Quentin, 2020

# General and computer vision lib
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot

def computeDerivative(img, sigmaX, sigmaY):
    # blurr the image
    img = cv2.GaussianBlur(img, ksize=(0,0), sigmaX=sigmaX, sigmaY=sigmaY)
    # create filter for derivative calulation
    dxFilter = np.array([[1],[0],[-1]])
    dyFilter = np.array([[1,0,-1]])
    dxxFilter = np.array([[1],[-2],[1]])
    dyyFilter = np.array([[1,-2,1]])
    dxyFilter = np.array([[1,-1],[-1,1]])
    # compute derivative
    dx = cv2.filter2D(img,-1, dxFilter)
    dy = cv2.filter2D(img,-1, dyFilter)
    dxx = cv2.filter2D(img,-1, dxxFilter)
    dyy = cv2.filter2D(img,-1, dyyFilter)
    dxy = cv2.filter2D(img,-1, dxyFilter)
    return dx, dy, dxx, dyy, dxy

def computeMagnitude(dxx, dyy):
    # convert to float
    dxx = dxx.astype(float)
    dyy = dyy.astype(float)
    # calculate magnitude and angle
    mag = cv2.magnitude(dxx, dyy)
    phase = mag*180./np.pi
    return mag, phase

def computeHessian(dx, dy, dxx, dyy, dxy):
    # create empty list
    point=[]
    direction=[]
    value=[]
    # for the all image
    for x in range(0, img.shape[1]): # column
        for y in range(0, img.shape[0]): # line
            # if superior to certain threshold
            if dxy[y,x] > 0:
                # compute local hessian
                hessian = np.zeros((2,2))
                hessian[0,0] = dxx[y,x]
                hessian[0,1] = dxy[y,x]
                hessian[1,0] = dxy[y,x]
                hessian[1,1] = dyy[y,x]
                # compute eigen vector and eigne value
                ret, eigenVal, eigenVect = cv2.eigen(hessian)
                if np.abs(eigenVal[0,0]) >= np.abs(eigenVal[1,0]):
                    nx = eigenVect[0,0]
                    ny = eigenVect[0,1]
                else:
                    nx = eigenVect[1,0]
                    ny = eigenVect[1,1]
                # calculate denominator for the taylor polynomial expension
                denom = dxx[y,x]*nx*nx + dyy[y,x]*ny*ny + 2*dxy[y,x]*nx*ny
                # verify non zero denom
                if denom != 0:
                    T = -(dx[y,x]*nx + dy[y,x]*ny)/denom
                    # update point
                    if np.abs(T*nx) <= 0.5 and np.abs(T*ny) <= 0.5:
                        point.append((x,y))
                        direction.append((nx,ny))
                        value.append(np.abs(dxy[y,x]+dxy[y,x]))
    return point, direction, value

# resize, grayscale and blurr
img = cv2.imread("im0.png")
img = cv2.cvtColor(img, cv2.COLOR_BGR2RGB)
img = cv2.resize(img, (640,480))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
# compute derivative
dx, dy, dxx, dyy, dxy = computeDerivative(gray_img, 1.1, 1.1)
normal, phase = computeMagnitude(dxx, dyy)
pt, dir, val = computeHessian(dx, dy, dxx, dyy, dxy)

# take the first n max value
nMax = 10000
idx = np.argsort(val)
idx = idx[::-1][:nMax]
# plot resulting point
for i in range(0, len(idx)):
    img = cv2.circle(img, (pt[idx[i]][0], pt[idx[i]][1]), 1, (255, 0, 0), 1)
# plot the result
plt.imshow(dx)
plt.show()
plt.imshow(dy)
plt.show()
plt.imshow(dxx)
plt.show()
plt.imshow(dyy)
plt.show()
plt.imshow(dxy)
plt.show()
plt.imshow(normal)
plt.show()
plt.imshow(phase)
plt.show()
plt.imshow(img)
plt.show()
