import csv
import distutils
import os
from distutils.dir_util import mkpath
from functools import partial

import shutil


class OutputWriter:
    cleaned_dir = "cleaned/"
    multi_faces_dir = "multi_faces/"
    outlier_faces_dir = "outlier_faces/"
    small_faces_dir = "small_faces/"
    no_faces_dir = "no_faces/"
    embeddings = "embeddings/"

    def __init__(self, output_dir):
        build_path = partial(os.path.join, output_dir)

        self.cleaned_dir = build_path(self.cleaned_dir)
        self.multi_faces_dir = build_path(self.multi_faces_dir)
        self.outlier_faces_dir = build_path(self.outlier_faces_dir)
        self.small_faces_dir = build_path(self.small_faces_dir)
        self.no_faces_dir = build_path(self.no_faces_dir)
        self.embeddings = build_path(self.embeddings)

        mkpath(self.cleaned_dir)
        mkpath(self.multi_faces_dir)
        mkpath(self.outlier_faces_dir)
        mkpath(self.small_faces_dir)
        mkpath(self.no_faces_dir)
        mkpath(self.embeddings)

    def _get_dst_path(self, src, dst_root):
        path = os.path.normpath(src)
        items = path.split(os.sep)
        # person_name/filename
        items = items[-2:]
        return os.path.join(dst_root, os.sep.join(items))

    def __copy_file(self, src, dst_dir):
        dst = self._get_dst_path(src, dst_dir)
        distutils.dir_util.mkpath(os.path.dirname(dst))
        shutil.copyfile(src, dst)

    def copy_to_cleaned(self, src):
        self.__copy_file(src, self.cleaned_dir)

    def copy_to_multi_faces(self, src):
        self.__copy_file(src, self.multi_faces_dir)

    def copy_to_outlier_faces(self, src):
        self.__copy_file(src, self.outlier_faces_dir)

    def copy_to_small_faces(self, src):
        self.__copy_file(src, self.small_faces_dir)

    def copy_to_no_faces(self, src):
        self.__copy_file(src, self.no_faces_dir)

    def is_embeddings_exist(self, src):
        file_path = os.path.join(self.embeddings, os.path.basename(src) + ".csv")
        return os.path.exists(file_path)

    def save_embeddings(self, src, data):
        file_path = os.path.join(self.embeddings, os.path.basename(src) + ".csv")

        file_handler = open(file_path, 'wt')
        try:
            writer = csv.writer(file_handler)
            for rec in data:
                writer.writerow(([rec[0], rec[1], rec[2]] + rec[3]))
        finally:
            file_handler.close()