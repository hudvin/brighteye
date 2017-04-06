
import glob
import os

import dlib
from skimage import io
from sklearn.externals import joblib
from sklearn.preprocessing import LabelEncoder

import warnings
warnings.filterwarnings("ignore", category=DeprecationWarning)

model = joblib.load("/tmp/model.pkl")

detector = dlib.get_frontal_face_detector()
sp = dlib.shape_predictor("/home/vbartko/projects/my/dlib/lib/models/shape_predictor_68_face_landmarks.dat")
facerec = dlib.face_recognition_model_v1("/home/vbartko/projects/my/dlib/lib/models/dlib_face_recognition_resnet_model_v1.dat")

input_file = "/mnt/hdd_ext/lab/datasets/facerecog/800_faces/benchmark/validation/14th_Dalai_Lama/68.jpg"

#print labels


work_dir = "/tmp"
print("Loading embeddings.")
fname = "{}/embeddings.csv".format(work_dir)
import pandas as pd
data = pd.read_csv(fname, header=None).as_matrix()
labels = data[:, 0]
le = LabelEncoder().fit(labels)

persons_dirs = glob.glob("/mnt/hdd_ext/lab/datasets/facerecog/800_faces/benchmark/validation" + "/*")
persons_files = map(lambda dir: (dir, glob.glob(dir + "/*")), persons_dirs)
total_counter = 0
wrong_counter = 0

import numpy as np

for person_dir, person_files in persons_files:
    print("total counter: %s, wrong counter: %s" % (total_counter, wrong_counter))
    person_embeddings = []
    for person_file in person_files:
        total_counter += 1
        input_file = person_file
        img = io.imread(input_file)
        dets = detector(img, 1)
        if len(dets) > 0:
            dets = dets[0]
            shape = sp(img, dets)
            face_descriptor = list(facerec.compute_face_descriptor(img, shape))
            label_inx = model.predict(face_descriptor)

            prediction_probs = model.predict_proba(face_descriptor)
            best_n =np.argsort(prediction_probs, axis=1)[0][-1:20]
            best_labels = [le.inverse_transform(n) for n in best_n]

            label = le.inverse_transform(label_inx)
            if  os.path.basename(person_dir) not in best_labels:
                print input_file, label
                wrong_counter+=1
        else:
            print "sorry"

