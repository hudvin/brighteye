#!/bin/bash
mkdir -p ../dlib_models
wget -qO-  http://dlib.net/files/shape_predictor_68_face_landmarks.dat.bz2 | bunzip2 > ../dlib_models/shape_predictor_68_face_landmarks.dat