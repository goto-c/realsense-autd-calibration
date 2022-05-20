import numpy as np
import cv2
import glob
import matplotlib.pyplot as plt

criteria = (cv2.TERM_CRITERIA_EPS + cv2.TERM_CRITERIA_MAX_ITER, 30, 0.001)

# prepare object points, like (0,0,0), (1,0,0), (2,0,0) ....,(10-1,7-1,0)
objp = np.zeros((27*18,3), np.float32)
objp[:,:2] = np.mgrid[0:27,0:18].T.reshape(-1,2)

objpoints = [] # 3d point in real world space
imgpoints = [] # 2d points in image plane.

images = sorted(glob.glob('calib_good_right/calib*.png'))
print(images)

for i, fname in enumerate(images):
    img = cv2.imread(fname)
    gray = cv2.cvtColor(img,cv2.COLOR_BGR2GRAY)
    ret, corners = cv2.findChessboardCorners(gray, (27,18),None)

    if ret == True:
        objpoints.append(objp)
        corners2 = cv2.cornerSubPix(gray,corners,(11,11),(-1,-1),criteria)
        imgpoints.append(corners2)
        img = cv2.drawChessboardCorners(img, (27,18), corners2,ret)
        if i==0: plt.imshow(img[...,::-1]); plt.show()

ret, mtx, dist, rvecs, tvecs = cv2.calibrateCamera(objpoints, imgpoints, gray.shape[::-1],None,None)
print("ret : ", ret)
print("mat : ", mtx)
print("dist : ", dist)
np.save("camera_right/mtx.npy", mtx); np.save("camera_right/dist.npy", dist)