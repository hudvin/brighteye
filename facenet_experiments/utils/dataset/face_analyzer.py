import os

import cv2
import dlib


def get_abs_path(current_dir, relative_path):
    return os.path.abspath(current_dir + relative_path)


class Errors:
    MANY_FACES = 10
    SMALL_FACE = 11
    OUTLIER = 20
    BW_IMAGE = 30
    NO_FACES = 40
    NO_ERROR = None


class BWFilter:
    def filter(self, cv_image):
        b, g, r = cv2.split(cv_image.image)
        is_grey = (cv2.absdiff(b, g).sum() == 0 and cv2.absdiff(b, r).sum() == 0)
        return is_grey


class FaceFilter:
    def __init__(self, face_detector):
        self.face_detector = face_detector

    def filter(self, cv_image, min_width, min_height):
        faces = self.face_detector.find_faces(cv_image)
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
            get_abs_path(current_dir, "/../../dlib_models/shape_predictor_68_face_landmarks.dat"))
        self.face_recognizer = dlib.face_recognition_model_v1(
            get_abs_path(current_dir, "/../../dlib_models/dlib_face_recognition_resnet_model_v1.dat"))

    def get_embeddings(self, cv_image):
        image = cv_image.image
        if len(image.shape) == 3:
            if image.shape[2] == 4:
                # drop alpha channel
                image = image[:, :, :3]
            detected_faces = self.detector(image, 1)
            shape = self.shape_predictor(image, detected_faces[0])
            face_descriptor = list(self.face_recognizer.compute_face_descriptor(image, shape))
            return None, face_descriptor


class FaceDetector:
    def __init__(self):
        self.face_detector = dlib.get_frontal_face_detector()

    def find_faces(self, cv_image):
        detection_results = self.face_detector(cv_image.image, 1)
        faces = []
        for i, face_rect in enumerate(detection_results):
            print("Detection {}: Left: {} Top: {} Right: {} Bottom: {}".format(
                i, face_rect.left(), face_rect.top(), face_rect.right(), face_rect.bottom()))
            faces.append({
                "top_left": (face_rect.left(), face_rect.top()),
                "bottom_right": (face_rect.right(), face_rect.bottom()),
                "width": face_rect.right() - face_rect.left(), "height": face_rect.bottom() - face_rect.top()})
        return faces
