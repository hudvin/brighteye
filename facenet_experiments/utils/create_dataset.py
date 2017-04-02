import argparse
import glob
import sys


def main(args):
    persons_dirs = glob.glob(args.input_dataset_dir + "/*")
    persons_files = map(lambda dir: (dir, glob.glob(dir + "/*")), persons_dirs)

    person_counter = 0
    for person_dir, person_files in persons_files:
        person_counter += 1
        file_counter = 0
        for persons_file in persons_files:
            #get number of faces
            #align face
            #get eye distance
            print persons_file
            file_counter += 1
            if file_counter >= args.photos_per_person:
                break
        print("person counter: %s" % person_counter)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('input_dataset_dir', type=str,
        help='Directory containing dirs with photos of persons ')
    #parser.add_argument('output_dataset_dir', type=str,
    #                    help='Directory to store processed images grouped by person name')
    #parser.add_argument('min_eye_distance', type=int, help='Min distance between eyes')
    #parser.add_argument('max_eye_distance', type=int, help='Min distance between eyes')
    parser.add_argument('photos_per_person', type=int, help='Number of photos per each person')

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))