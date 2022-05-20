import cv2
import numpy as np

marker_length = 0.10

XYZ = []
RPY = []
V_x = []
V_y = []
V_z = []

aruco_dict = cv2.aruco.getPredefinedDictionary(cv2.aruco.DICT_4X4_50)
mtx = np.load("camera/camera_left/mtx.npy")
dist = np.load("camera/camera_left/dist.npy")

frame = cv2.imread("aruco_left.png") 
frame = frame[...,::-1]  # BGR2RGB
corners, ids, _ = cv2.aruco.detectMarkers(frame, aruco_dict)

if len(corners) == 0:
    print("No corner detected !")

rvec, tvec, _ = cv2.aruco.estimatePoseSingleMarkers(corners, marker_length, mtx, dist)

R = cv2.Rodrigues(rvec)[0]  # 回転ベクトル -> 回転行列
R_T = R.T
T = tvec[0].T
print("R : ", R)
print("T : ", T)

f = open('mat1.txt', 'w')
for i in range(3):
    for j in range(3):
        f.write(str(R[i][j]))
        f.write("\n")
    f.write(str(T[i][0]))
    f.write("\n")
f.write(str(0) + "\n" + str(0) + "\n" + str(0) + "\n" + str(1))
f.close()

xyz = np.dot(R_T, - T).squeeze()
XYZ.append(xyz)

rpy = np.deg2rad(cv2.RQDecomp3x3(R_T)[0])
RPY.append(rpy)

V_x.append(np.dot(R_T, np.array([1,0,0])))
V_y.append(np.dot(R_T, np.array([0,1,0])))
V_z.append(np.dot(R_T, np.array([0,0,1])))

# ---- 描画
print("frame : ", frame.shape)
frame = np.ascontiguousarray(frame, dtype=np.uint8)
cv2.aruco.drawDetectedMarkers(frame, corners, ids, (0,255,255))
cv2.aruco.drawAxis(frame, mtx, dist, rvec, tvec, marker_length/2)
cv2.imshow('frame', frame)
cv2.waitKey(0)
# ----

cv2.destroyAllWindows()