import argparse
import glob
import multiprocessing
import sys
import traceback
from functools import partial

from dataset.centroid_filter import CentroidFilter
from dataset.face_analyzer import FaceDetector, FaceFilter, Errors, BWFilter
from dataset.output_writer import OutputWriter

from utils.dataset.cv_image import CvImage


def main(args):
    print(args)
    output_dirs = OutputWriter(args.output_dir)

    persons_dirs = glob.glob(args.input_dir + "/*")
    persons_files = map(lambda dir: (dir,
                                     filter(lambda filepath: filepath.endswith((".jpg", ".png", ".jpeg")),
                                            glob.glob(dir + "/*"))),
                        persons_dirs)
    if args.max_persons != -1:
        persons_files = persons_files[:args.max_persons]

    pool = multiprocessing.Pool(processes=args.threads)
    process_persons_args = partial(process_persons, args, output_dirs)
    pool.map(process_persons_args, persons_files)


def process_persons(args, output_dirs, person_files):
    try:
        min_face_widht, min_face_height = map(lambda x: int(x), args.min_face_dims.split("x"))
        (dir, files) = person_files
        if not output_dirs.is_embeddings_exist(dir):
            face_filter = FaceFilter(FaceDetector())
            bw_filter = BWFilter()
            centroid_filter = CentroidFilter()
            single_face_images = []
            cv_images = [CvImage(file) for file in files]
            for cv_image in cv_images:
                print "processing %s" % cv_image
                if args.skip_bw and bw_filter.filter(cv_image):
                    print "image %s is bw" % cv_image.src
                    output_dirs.copy_to_bw(cv_image.src)
                else:
                    error, result = face_filter.filter(cv_image, min_face_widht, min_face_height)
                    if not error:
                        single_face_images.append(cv_image)
                    else:
                        if error == Errors.MANY_FACES:
                            output_dirs.copy_to_multi_faces(cv_image.src)
                            pass
                        elif error == Errors.SMALL_FACE:
                            output_dirs.copy_to_small_faces(cv_image.src)
                        elif error == Errors.NO_FACES:
                            output_dirs.copy_to_no_faces(cv_image.src)
            print "generating embeddings, apply centroid filtering"
            if single_face_images:
                bad, good = centroid_filter.filter(single_face_images, 0.5)

                for record in bad:
                    output_dirs.copy_to_outlier_faces(record[0])

                max = args.max_photos if args.max_photos != -1 else len(good)
                good = good[:max]
                for record in good:
                    output_dirs.copy_to_cleaned(record[0])
                output_dirs.save_embeddings(dir, good)
    except Exception:
        print(traceback.format_exc())


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--input-dir', type=str, help='Directory containing dirs with photos of persons', required=True)
    parser.add_argument('--output-dir', type=str, help='Directory to store processed images grouped by person name', required=True)
    parser.add_argument('--max-photos', type=int, help='Number of photos per each person', default=-1)
    parser.add_argument('--max-persons', type=int, help='Max number of persons', default=-1)
    parser.add_argument('--min-face-dims', type=str, help='Min dims(widthxheight) of face', default="100x100")
    parser.add_argument('--threads', type=int, help='Num of threads to use', default=6)
    parser.add_argument('--skip-bw', type=bool, help='Skip grayscale images', default=True)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
