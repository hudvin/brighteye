import glob
import os
import unittest

from face_analyzer import EmbeddingsExtractor, FaceDetector, FaceFilter, CentroidFilter

faces_800_root = os.environ['faces_800']


class FaceAnalyzerTest(unittest.TestCase):
    @unittest.skip
    def test_embeddings_extractor(self):
        embeddings_extractor = EmbeddingsExtractor()
        result, error = embeddings_extractor.get_embeddings("facenet_experiments/images/portman1.jpg")
        print result, error

    def test_outliers_detector(self):
        face_filter = FaceFilter(FaceDetector())
        persons_dir = glob.glob(faces_800_root + "David Hidalgo/*.jpg")

        files = []
        for person_file in persons_dir:
            error, result = face_filter.filter(person_file, 100, 100)
            if result:
                files.append(person_file)
                print person_file
            else:
                print error

        centroid_filter = CentroidFilter()
        bad, good = centroid_filter.filter(files, 0.5)
        print "bad photos: \n", bad
        print "good photos:\n", good


if __name__ == '__main__':
    unittest.main()
