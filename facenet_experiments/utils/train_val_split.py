import argparse
import glob
import os
import sys
from distutils.dir_util import mkpath

import shutil


def main(args):
    persons_dirs = glob.glob(args.input_dataset_dir + "/*")
    persons_files = map(lambda dir: (dir, glob.glob(dir + "/*")), persons_dirs)

    train_persons_files = map(lambda (dir, files): (dir, files[0:int(len(files)*(1-args.validation_ration))-1]), persons_files)
    validation_person_files = map(lambda (dir, files): (dir, files[-int(len(files)*args.validation_ration)+1:]), persons_files)

    create_set(train_persons_files, args.output_dataset_dir + "/" + "train")
    create_set(validation_person_files, args.output_dataset_dir + "/" + "validation")


def create_set(persons_files, output_dir):
    for (dir, files) in persons_files:
        person_dir = os.path.basename(dir)
        print person_dir
        validation_person_dir = output_dir + "/" + person_dir
        mkpath(validation_person_dir)
        for person_file in files:
            print person_file
            file_basename = os.path.basename(person_file)
            shutil.copyfile(person_file, validation_person_dir + "/" + file_basename)


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dataset_dir', type=str, help='Directory containing dirs with photos of persons ')
    parser.add_argument('output_dataset_dir', type=str, help='Directory to store processed images grouped by person name')
    parser.add_argument('validation_ration', type=float, help='Validation ratio')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))