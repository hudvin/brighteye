# based on https://github.com/mpatacchiola/deepgaze/blob/master/examples/ex_dlib_pnp_head_pose_estimation_video.py

import numpy
import cv2
import dlib

# Antropometric constant values of the human head.
# Found on wikipedia and on:
# "Head-and-Face Anthropometric Survey of U.S. Respirator Users"
#
# X-Y-Z with X pointing forward and Y on the left.
# The X-Y-Z coordinates used are like the standard
# coordinates of ROS (robotic operative system)
P3D_RIGHT_SIDE = numpy.float32([-100.0, -77.5, -5.0])  # 0
P3D_GONION_RIGHT = numpy.float32([-110.0, -77.5, -85.0])  # 4
P3D_MENTON = numpy.float32([0.0, 0.0, -122.7])  # 8
P3D_GONION_LEFT = numpy.float32([-110.0, 77.5, -85.0])  # 12
P3D_LEFT_SIDE = numpy.float32([-100.0, 77.5, -5.0])  # 16
P3D_FRONTAL_BREADTH_RIGHT = numpy.float32([-20.0, -56.1, 10.0])  # 17
P3D_FRONTAL_BREADTH_LEFT = numpy.float32([-20.0, 56.1, 10.0])  # 26
P3D_SELLION = numpy.float32([0.0, 0.0, 0.0])  # 27
P3D_NOSE = numpy.float32([21.1, 0.0, -48.0])  # 30
P3D_SUB_NOSE = numpy.float32([5.0, 0.0, -52.0])  # 33
P3D_RIGHT_EYE = numpy.float32([-20.0, -65.5, -5.0])  # 36
P3D_RIGHT_TEAR = numpy.float32([-10.0, -40.5, -5.0])  # 39
P3D_LEFT_TEAR = numpy.float32([-10.0, 40.5, -5.0])  # 42
P3D_LEFT_EYE = numpy.float32([-20.0, 65.5, -5.0])  # 45
# P3D_LIP_RIGHT = numpy.float32([-20.0, 65.5,-5.0]) #48
# P3D_LIP_LEFT = numpy.float32([-20.0, 65.5,-5.0]) #54
P3D_STOMION = numpy.float32([10.0, 0.0, -75.0])  # 62

# The points to track
# These points are the ones used by PnP
# to estimate the 3D pose of the face
TRACKED_POINTS = (0, 4, 8, 12, 16, 17, 26, 27, 30, 33, 36, 39, 42, 45, 62)

image_path = "/home/kontiki/projects/work/level11/notebooks/portman4.jpg"
# Read Image
im = cv2.imread(image_path)

height, width, channels = im.shape

# Obtaining the CAM dimension
cam_w = int(width)
cam_h = int(height)

# Defining the camera matrix.
# To have better result it is necessary to find the focal
# lenght of the camera. fx/fy are the focal lengths (in pixels)
# and cx/cy are the optical centres. These values can be obtained
# roughly by approximation, for example in a 640x480 camera:
# cx = 640/2 = 320
# cy = 480/2 = 240
# fx = fy = cx/tan(60/2 * pi / 180) = 554.26
c_x = cam_w / 2
c_y = cam_h / 2
f_x = c_x / numpy.tan(60 / 2 * numpy.pi / 180)
f_y = f_x

# Estimated camera matrix values.
camera_matrix = numpy.float32([[f_x, 0.0, c_x],
                               [0.0, f_y, c_y],
                               [0.0, 0.0, 1.0]])

print("Estimated camera matrix: \n" + str(camera_matrix) + "\n")


# Distortion coefficients
camera_distortion = numpy.float32([0.0, 0.0, 0.0, 0.0, 0.0])

# Distortion coefficients estimated by calibration
# camera_distortion = numpy.float32([ 0.06232237, -0.41559805,  0.00125389, -0.00402566,  0.04879263])


# This matrix contains the 3D points of the
# 11 landmarks we want to find. It has been
# obtained from antrophometric measurement
# on the human head.
landmarks_3D = numpy.float32([P3D_RIGHT_SIDE,
                              P3D_GONION_RIGHT,
                              P3D_MENTON,
                              P3D_GONION_LEFT,
                              P3D_LEFT_SIDE,
                              P3D_FRONTAL_BREADTH_RIGHT,
                              P3D_FRONTAL_BREADTH_LEFT,
                              P3D_SELLION,
                              P3D_NOSE,
                              P3D_SUB_NOSE,
                              P3D_RIGHT_EYE,
                              P3D_RIGHT_TEAR,
                              P3D_LEFT_TEAR,
                              P3D_LEFT_EYE,
                              P3D_STOMION])

PREDICTOR_PATH = "/home/kontiki/projects/work/level11/notebooks/shape_predictor_68_face_landmarks.dat"
predictor = dlib.shape_predictor(PREDICTOR_PATH)
detector = dlib.get_frontal_face_detector()

dets = detector(im, 1)
det = dets[0]

face_x1 = det.left()
face_y1 = det.top()
face_x2 = det.right()
face_y2 = det.bottom()

cv2.rectangle(im,
              (face_x1, face_y1),
              (face_x2, face_y2),
              (0, 255, 0),
              2)

landmarks = [(p.x, p.y) for p in predictor(im, det).parts()]


landmarks = [[i, pos] for i, pos in enumerate(landmarks) if i in TRACKED_POINTS]

for i, pos in landmarks:
    cv2.circle(im, (pos[0], pos[1]), 2, (0, 0, 255), -1)
    print i, pos

landmarks_2D = numpy.float32([(pos[0], pos[1]) for i, pos in landmarks])
print landmarks_2D

retval, rvec, tvec = cv2.solvePnP(landmarks_3D,
                                  landmarks_2D,
                                  camera_matrix, camera_distortion)

axis = numpy.float32([[50, 0, 0],
                      [0, 50, 0],
                      [0, 0, 50],
                      [21.1, 0.0, -48.0]
                      ])


landmarks_3D_2D = numpy.delete(landmarks_3D, 2, 1)
M, _ = cv2.findHomography(landmarks_3D_2D,landmarks_2D)


imgpts, jac = cv2.projectPoints(axis, rvec, tvec, camera_matrix, camera_distortion)

# Drawing the three axis on the image frame.
# The opencv colors are defined as BGR colors such as:
# (a, b, c) >> Blue = a, Green = b and Red = c
# Our axis/color convention is X=R, Y=G, Z=B
sellion_xy = (landmarks_2D[7][0], landmarks_2D[7][1])
cv2.line(im, sellion_xy, tuple(imgpts[1].ravel()), (0, 255, 0), 3)  # GREEN
cv2.line(im, sellion_xy, tuple(imgpts[2].ravel()), (255, 0, 0), 3)  # BLUE
cv2.line(im, sellion_xy, tuple(imgpts[0].ravel()), (0, 0, 255), 3)  # RED

#cv2.circle(im, tuple(imgpts[3]), 10,  (0,0,255))

cv2.circle(im, tuple(imgpts[3].ravel()) , 5, (0, 0, 255), -1)

#cv2.namedWindow("face2", cv2.WINDOW_NORMAL)
cv2.imshow('face2', cv2.warpPerspective(im, M,im.shape[1::-1]))


cv2.imshow('face', im)
cv2.waitKey(-1)
