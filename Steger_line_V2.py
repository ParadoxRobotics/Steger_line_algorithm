# Steger algorithm for edge/line extraction
# Author : Munch Quentin, 2020

# General and computer vision lib
import numpy as np
import cv2
from matplotlib import pyplot as plt
from matplotlib import pyplot

def computeDerivative(img):
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


def nonMaxSuppression(dxx, dyy, dxy):
    # compute magnitude of each derivative
    dxx = dxx.astype(float)
    dyy = dyy.astype(float)
    magnitude = cv2.magnitude(dxx, dyy)
    plt.imshow(magnitude)
    plt.show()
    angle = magnitude*180./np.pi
    angle[angle < 0] += 180
    # get derivative shape
    W, H = dxy.shape
    # create an empty gradient map
    mapGrad = np.zeros((H,W), dtype=np.int32)

    for i in range(0,H):
        for j in range(0,W):
            q = 255
            r = 255
           #angle 0
            if (0 <= angle[i,j] < 22.5) or (157.5 <= angle[i,j] <= 180):
                # position
                qi = i
                ri = i
                qj = i+1
                rj = j-1
                # verify border in i
                if qi < 0:
                    qi = 0
                if qi > H-1:
                    qi = H-1
                if ri < 0:
                    ri = 0
                if ri > H-1:
                    ri = H-1
                # verify border in j
                if qj < 0:
                    qj = 0
                if qj > W-1:
                    qj = W-1
                if rj < 0:
                    rj = 0
                if rj > W-1:
                    rj = W-1
                # apply
                q = dxy[qi, qj]
                r = dxy[ri, rj]
            #angle 45
            elif (22.5 <= angle[i,j] < 67.5):
                # position
                qi = i+1
                ri = i-1
                qj = j-1
                rj = j+1
                # verify border in i
                if qi < 0:
                    qi = 0
                if qi > H-1:
                    qi = H-1
                if ri < 0:
                    ri = 0
                if ri > H-1:
                    ri = H-1
                # verify border in j
                if qj < 0:
                    qj = 0
                if qj > W-1:
                    qj = W-1
                if rj < 0:
                    rj = 0
                if rj > W-1:
                    rj = W-1
                print(qi, qj, H-1)
                print(ri, rj, W-1)
                # apply
                q = dxy[qi, qj]
                r = dxy[ri, rj]
            #angle 90
            elif (67.5 <= angle[i,j] < 112.5):
                qi = i+1
                qj = j
                ri = i-1
                rj = j
                # verify border in i
                if qi < 0:
                    qi = 0
                if qi > H-1:
                    qi = H-1
                if ri < 0:
                    ri = 0
                if ri > H-1:
                    ri = H-1
                # verify border in j
                if qj < 0:
                    qj = 0
                if qj > W-1:
                    qj = W-1
                if rj < 0:
                    rj = 0
                if rj > W-1:
                    rj = W-1
                # apply
                q = dxy[qi, qj]
                r = dxy[ri, rj]
            #angle 135
            elif (112.5 <= angle[i,j] < 157.5):
                qi = i-1
                qj = j-1
                ri = i+1
                rj = j+1
                # verify border in i
                if qi <= 0:
                    qi = 0
                if qi > H-1:
                    qi = H-1
                if ri < 0:
                    ri = 0
                if ri > H-1:
                    ri = H-1
                # verify border in j
                if qj < 0:
                    qj = 0
                if qj > W-1:
                    qj = W-1
                if rj < 0:
                    rj = 0
                if rj > W-1:
                    rj = W-1
                # apply
                q = dxy[qi, qj]
                r = dxy[ri, rj]

            if (dxy[i,j] >= q) and (dxy[i,j] >= r):
                mapGrad[i,j] = dxy[j,i]
            else:
                mapGrad[i,j] = 0

    return mapGrad


def computeHessian(img, dx, dy, dxx, dyy, dxy):
    point=[]
    # for the all image
    for x in range(0, img.shape[1]): # column
        for y in range(0, img.shape[0]): # line
            # if superior to certain threshold
            if img[y,x] > 10:
                # compute local hessian
                hessian = np.zeros((2,2))
                hessian[0,0] = dxx[y,x]
                hessian[0,1] = dxy[y,x]
                hessian[1,0] = dxy[y,x]
                hessian[1,1] = dyy[y,x]
                # compute eigen vector and eigne value
                ret, eigenVal, eigenVect = cv2.eigen(hessian)
                if np.abs(eigenVal[0,0]) <= np.abs(eigenVal[1,0]):
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
    return point

# resize, grayscale and blurr
img = cv2.imread("im0.png")
img = cv2.resize(img, (320,240))
gray_img = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)
gray_gauss_img = cv2.GaussianBlur(gray_img, ksize=(0,0), sigmaX=1, sigmaY=1)
# compute derivative
dx, dy, dxx, dyy, dxy = computeDerivative(gray_gauss_img)
# non-max suppresion
dxy = nonMaxSuppression(dxx, dyy, dxy)
# compute relevant point
pt = computeHessian(gray_gauss_img, dx, dy, dxx, dyy, dxy)
# plot points
for i in range(0, len(pt)):
    img = cv2.circle(img, (pt[i][0], pt[i][1]), 1, (255, 0, 0), 1)

plt.imshow(dxy)
plt.show()
plt.imshow(img)
plt.show()
