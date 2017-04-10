import dlib

class EmbeddingsExtractor:
    pass


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
