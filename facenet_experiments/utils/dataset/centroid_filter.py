from face_analyzer import EmbeddingsExtractor
import numpy as np
import os
from scipy.spatial import distance


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
                    data_list.append(np.array([person_file] + result))
                else:
                    print "can't extract embeddings for %s" % person_file, result
            except Exception as e:
                print e, person_file

        # convert to np array
        data_list = np.array(data_list)
        # extract embeddings column and convert it to float
        embeddings = np.delete(data_list, 0, 1).astype(float)
        # get labels column, by some reason last part is required
        cv_images = data_list[:, [0]][:, 0]
        # calculate centroid
        centroid = np.array([np.mean(row) for row in embeddings.T])
        # convert it to row
        centroid = centroid.T
        # 128 elements in row
        distances = np.array([distance.euclidean(centroid, row) for row in embeddings])

        # label - path - embedidngs
        labels_distance_embeddings = [
            (cv_image.src, os.path.normpath(cv_image.src).split(os.sep)[-2:-1][0], centroid_distance, embedding.tolist())
            for cv_image, centroid_distance, embedding in zip(cv_images, distances, embeddings)
            ]
        labels_distance_embeddings.sort(key=lambda rec: rec[2])
        bad = [rec for rec in labels_distance_embeddings if rec[2] > threshold]
        good = [rec for rec in labels_distance_embeddings if rec[2] <= threshold]
        return bad, good
