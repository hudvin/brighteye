#!/bin/bash
echo "downloading shape predictor model..."
wget -qO-  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 | bunzip2 > shape_predictor_68_face_landmarks.dat

echo "downloading face recognition model..."
wget -qO-  http://dlib.net/files/dlib_face_recognition_resnet_model_v1.dat.bz2 | bunzip2 > dlib_face_recognition_resnet_model_v1.dat

echo "done!"