import cv2
import dlib


class DlibLoader:
    def __init__(self):
        pass


class FaceAnalyzer:
    def __init__(self):
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
                "width": face_rect.right()-face_rect.left(), "height":  face_rect.bottom()-face_rect.top()})
        return faces

#
# class FaceAnalyzer:
#
#     def __init__(self, dlib_predictor_model):
#         dlib_predictor_file = os.path.join(
#             dlib_model_dir, "shape_predictor_68_face_landmarks.dat")
#         aligner = openface.AlignDlib(dlib_predictor_file)
#
#
# def get_angles():
#     pass
#
# def get_eye_distance():
#     pass
#
#
# def align_face():
#     pass
#
# def get_num_of_faces(img_file):
#     bgr_img = cv2.imread(img_file)
#     rgb_img = cv2.cvtColor(bgr_img, cv2.COLOR_BGR2RGB)
#     bbs = aligner.getAllFaceBoundingBoxes(rgb_img)
#     return len(bbs)
