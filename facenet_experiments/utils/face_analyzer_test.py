import unittest

from face_analyzer import EmbeddingsExtractor


class FaceAnalyzerTest(unittest.TestCase):

    def test_embeddings_extractor(self):
        embeddings_extractor = EmbeddingsExtractor()
        print embeddings_extractor.get_embeddings(["facenet_experiments/images/portman1.jpg"])


if __name__ == '__main__':
    unittest.main()