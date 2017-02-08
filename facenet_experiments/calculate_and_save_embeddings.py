"""Performs face alignment and calculates L2 distance between the embeddings of two images."""

# MIT License
#
# Copyright (c) 2016 David Sandberg
#
# Permission is hereby granted, free of charge, to any person obtaining a copy
# of this software and associated documentation files (the "Software"), to deal
# in the Software without restriction, including without limitation the rights
# to use, copy, modify, merge, publish, distribute, sublicense, and/or sell
# copies of the Software, and to permit persons to whom the Software is
# furnished to do so, subject to the following conditions:
#
# The above copyright notice and this permission notice shall be included in all
# copies or substantial portions of the Software.
#
# THE SOFTWARE IS PROVIDED "AS IS", WITHOUT WARRANTY OF ANY KIND, EXPRESS OR
# IMPLIED, INCLUDING BUT NOT LIMITED TO THE WARRANTIES OF MERCHANTABILITY,
# FITNESS FOR A PARTICULAR PURPOSE AND NONINFRINGEMENT. IN NO EVENT SHALL THE
# AUTHORS OR COPYRIGHT HOLDERS BE LIABLE FOR ANY CLAIM, DAMAGES OR OTHER
# LIABILITY, WHETHER IN AN ACTION OF CONTRACT, TORT OR OTHERWISE, ARISING FROM,
# OUT OF OR IN CONNECTION WITH THE SOFTWARE OR THE USE OR OTHER DEALINGS IN THE
# SOFTWARE.

from __future__ import absolute_import
from __future__ import division
from __future__ import print_function

import glob

from scipy import misc
import tensorflow as tf

import numpy as np
import sys
import os
import argparse
import facenet
import align.detect_face

np.set_printoptions(threshold=np.nan)


def main(args):
    aligned_images_root_dir = args.aligned_images_root_dir

    with tf.Graph().as_default():

        with tf.Session() as sess:

            # Load the model
            print('Model directory: %s' % args.model_dir)
            meta_file, ckpt_file = facenet.get_model_filenames(os.path.expanduser(args.model_dir))
            print('Metagraph file: %s' % meta_file)
            print('Checkpoint file: %s' % ckpt_file)
            facenet.load_model(args.model_dir, meta_file, ckpt_file)

            persons_dirs = glob.glob(aligned_images_root_dir + "/*")[0:50]
            persons_files = map(lambda dir: (dir, glob.glob(dir + "/*")), persons_dirs)
            counter = 0
            for person_dir, person_files in persons_files:
                counter +=1
                print("person counter: %s" % counter)
                images = load_images(person_files)
                # Get input and output tensors
                images_placeholder = tf.get_default_graph().get_tensor_by_name("input:0")
                embeddings = tf.get_default_graph().get_tensor_by_name("embeddings:0")

                # Run forward pass to calculate embeddings
                phase_train_placeholder = tf.get_default_graph().get_tensor_by_name("phase_train:0")
                feed_dict = {images_placeholder: images, phase_train_placeholder:False}

                person_name = os.path.basename(person_dir)
                emb = sess.run(embeddings, feed_dict=feed_dict)
                for em in emb:
                    em = em.astype(str)
                    em = np.insert(em, 0, person_name)
                    print(len(em[0]))
                    print(em[0])
                    with open("/tmp/embeddings.csv", 'a') as f_handle:
                        np.savetxt(f_handle, [em], fmt="%s", delimiter=", ")



def load_images(image_paths):
    nrof_samples = len(image_paths)
    img_list = [None] * nrof_samples
    for i in xrange(nrof_samples):
        img = misc.imread(image_paths[i])
        img_list[i] = img
    images = np.stack(img_list)
    return images


def parse_arguments(argv):
    parser = argparse.ArgumentParser()

    parser.add_argument('model_dir', type=str,
                        help='Directory containing the meta_file and ckpt_file')
    parser.add_argument('aligned_images_root_dir', type=str, help='Path to root dir with aligned images')
    parser.add_argument('--image_size', type=int,
                        help='Image size (height, width) in pixels.', default=160)
    parser.add_argument('--margin', type=int,
                        help='Margin for the crop around the bounding box (height, width) in pixels.', default=44)
    parser.add_argument('--gpu_memory_fraction', type=float,
                        help='Upper bound on the amount of GPU memory that will be used by the process.', default=1.0)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
