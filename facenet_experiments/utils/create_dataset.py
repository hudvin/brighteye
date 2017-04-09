import argparse
import glob
import os
import sys
from distutils.dir_util import mkpath

import shutil

import cv2
from face_analyzer import FaceDetector


def main(args):
    print(args)
    min_face_widht, min_face_height = map(lambda x: int(x), args.min_face_dims.split("x"))
    persons_dirs = glob.glob(args.input_dir + "/*")
    persons_files = map(lambda dir: (dir,
                                     filter(lambda filepath: filepath.endswith((".jpg", ".png", ".jpeg")),
                                            glob.glob(dir + "/*"))),
                        persons_dirs)
    if args.max_persons != -1:
        persons_files = persons_files[:args.max_persons]

    face_detector = FaceDetector()
    person_counter = 0
    for person_dir, person_files in persons_files:
        person_counter += 1
        file_counter = 0
        output_person_dir = args.output_dir + "/" + os.path.basename(person_dir)
        mkpath(output_person_dir)
        for person_file in person_files:
            print person_file
            faces = face_detector.find_faces(cv2.imread(person_file))
            if not faces:
                print "skipping %s, no faces found" % (person_file)
            elif len(faces) > 1:
                print "skipping %s, contains %s faces" % (person_file, len(faces))
            else:
                face = faces[0]
                face_width = face["width"]
                face_height = face["height"]
                if face_width >= min_face_widht and face_height >= min_face_height:
                    file_counter += 1
                    shutil.copyfile(person_file, output_person_dir + "/" + os.path.basename(person_file))
                    if args.max_photos != -1 and file_counter >= args.max_photos:
                        break
                else:
                    print "skipping %s, because face dims are too small - (%s x %s)" % (
                    person_file, face_width, face_height)
        print("person counter: %s" % person_counter)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str,
                        help='Directory containing dirs with photos of persons ')
    parser.add_argument('--output-dir', type=str,
                        help='Directory to store processed images grouped by person name')
    parser.add_argument('--max-photos', type=int, help='Number of photos per each person', default=-1)
    parser.add_argument('--max-persons', type=int, help='Max number of persons', default=-1)
    parser.add_argument('--min_face_dims', type=str, help='Min dims(widthxheight) of face', default="100x100")
    # parser.add_argument('--num-faces', type=int, help='Max number of faces on image', default=1)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
