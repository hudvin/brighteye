import cv2


class CvImage:
    def __init__(self, file_path):
        self.src = file_path
        self.image = cv2.imread(file_path)

    def __str__(self):
        return "cv2 image, path: %s" % self.src
