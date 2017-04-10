import os

import dlib
from skimage import io


def get_abs_path(current_dir, relative_path):
    return os.path.abspath(current_dir + relative_path)


class EmbeddingsExtractor:
    def __init__(self):
        self.detector = dlib.get_frontal_face_detector()
        current_dir = (os.path.dirname(os.path.realpath(__file__)))
        self.shape_predictor = dlib.shape_predictor(get_abs_path(current_dir, "/../dlib_models/shape_predictor_68_face_landmarks.dat"))
        self.face_recognizer = dlib.face_recognition_model_v1(get_abs_path(current_dir, "/../dlib_models/dlib_face_recognition_resnet_model_v1.dat"))

    def get_embeddings(self, person_files):
        embeddings = []
        for person_file in person_files:
            image = io.imread(person_file)
            detected_faces = self.detector(image, 1)
            shape = self.shape_predictor(image, detected_faces[0])
            face_descriptor = list(self.face_recognizer.compute_face_descriptor(image, shape))
            embeddings.append({"file": person_file, "embeddings": face_descriptor})
        return embeddings


class OutlierDetector:
    pass


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
