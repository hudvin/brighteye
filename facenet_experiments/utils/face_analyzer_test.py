import glob
import os
import unittest

from face_analyzer import EmbeddingsExtractor, FaceDetector, FaceFilter

faces_800_root = os.environ['faces_800']

class FaceAnalyzerTest(unittest.TestCase):

    @unittest.skip
    def test_embeddings_extractor(self):
        embeddings_extractor = EmbeddingsExtractor()
        print embeddings_extractor.get_embeddings(["facenet_experiments/images/portman1.jpg"])


    def test_outliers_detector(self):
        face_filter = FaceFilter(FaceDetector())
        persons_dir = glob.glob(faces_800_root +"14th Dalai Lama/*.jpg")
        embeddings_extractor = EmbeddingsExtractor()

        results = []
        for person_file in persons_dir:
            error, result = face_filter.filter(person_file, 100, 100)
            if result:
                results.append(embeddings_extractor.get_embeddings([person_file])[0])
                print person_file
            else:
                print error
        print results
        data_list = []
        for item in results:
            rec = []
            rec.append(item["file"])
            rec.extend(item["embeddings"])
            data_list.append(rec)
        print data_list



if __name__ == '__main__':
    unittest.main()