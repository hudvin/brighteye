import os

import cv2
import dlib
from os.path import basename
from skimage import io
import numpy as np
from scipy.spatial import distance


def get_abs_path(current_dir, relative_path):
    return os.path.abspath(current_dir + relative_path)


class Errors:
    MANY_FACES = 10
    SMALL_FACE = 11
    OUTLIER = 20
    BW_IMAGE = 30
    NO_FACES = 40
    none = None


class FaceFilter:
    def __init__(self, face_detector):
        self.face_detector = face_detector

    def filter(self, person_file, min_width, min_height):
        faces = self.face_detector.find_faces(cv2.imread(person_file))
        if not faces:
            return Errors.NO_FACES, None
        elif len(faces) > 1:
            return Errors.MANY_FACES, "image contains %s faces" % len(faces)
        else:
            face = faces[0]
            face_width = face["width"]
            face_height = face["height"]
            if face_width >= min_width and face_height >= min_height:
                return None, True
            else:
                return Errors.SMALL_FACE, "face dims are (%s x %s)" % (face_width, face_height)


class EmbeddingsExtractor:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        current_dir = (os.path.dirname(os.path.realpath(__file__)))
        self.shape_predictor = dlib.shape_predictor(
            get_abs_path(current_dir, "/../dlib_models/shape_predictor_68_face_landmarks.dat"))
        self.face_recognizer = dlib.face_recognition_model_v1(
            get_abs_path(current_dir, "/../dlib_models/dlib_face_recognition_resnet_model_v1.dat"))

    def get_embeddings(self, person_file):
        image = io.imread(person_file)
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                # drop alpha channel
                image = image[:, :, :3]
            detected_faces = self.detector(image, 1)
            shape = self.shape_predictor(image, detected_faces[0])
            face_descriptor = list(self.face_recognizer.compute_face_descriptor(image, shape))
            return None, {"file": person_file, "embeddings": face_descriptor},
        elif len(image.shape) == 2:
            return Errors.BW_IMAGE, "image is BW"


class CentroidFilter:
    def __init__(self):
        self.embeddings_extractor = EmbeddingsExtractor()

    def filter(self, person_files, threshold):
        data_list = []
        for person_file in person_files:
            # because it can take list of files as argument
            # so we pass list and get list
            try:
                error, result = self.embeddings_extractor.get_embeddings(person_file)
                if not error:
                    data_list.append(np.array([result["file"]] + result["embeddings"]))
                else:
                    print "can't extract embeddings for %s" % person_file, result
            except Exception as e:
                print e, person_file

        # convert to np array
        data_list = np.array(data_list)
        # extract embeddings column and convert it to float
        embeddings = np.delete(data_list, 0, 1).astype(float)
        # get labels column, by some reason last part is required
        labels = data_list[:, [0]][:, 0]
        # calculate centroid
        centroid = np.array([np.mean(row) for row in embeddings.T])
        # convert it to row
        centroid = centroid.T
        # 128 elements in row
        distances = np.array([distance.euclidean(centroid, row) for row in embeddings])

        labels_distance_embeddings = [
            (label, os.path.normpath(label).split(os.sep)[-2:-1][0], centroid_distance, embedding.tolist())
            for label, centroid_distance, embedding in zip(labels, distances, embeddings)
            ]
        labels_distance_embeddings.sort(key=lambda rec: rec[2])
        bad = [rec for rec in labels_distance_embeddings if rec[2] > threshold]
        good = [rec for rec in labels_distance_embeddings if rec[2] <= threshold]
        return bad, good


class FaceDetector:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()

    def find_faces(self, image_bytes):
        detection_results = self.face_detector(image_bytes, 1)
        faces = []
        for i, face_rect in enumerate(detection_results):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
            faces.append({
                "top_left": (face_rect.left(), face_rect.top()),
                "bottom_right": (face_rect.right(), face_rect.bottom()),
                "width": face_rect.right() - face_rect.left(), "height": face_rect.bottom() - face_rect.top()})
        return faces
