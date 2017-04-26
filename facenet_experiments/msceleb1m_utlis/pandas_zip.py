import argparse
import sys
import zipfile
from distutils.dir_util import mkpath


def main(args):
    print(args)
    zip_archive = zipfile.ZipFile(args.msceleb_zip)
    tsv_file = zip_archive.open('MsCelebV1-ImageThumbnails.tsv', "r")

    name_dirs = []
    total_counter = 0

    sub_dir_index = 0
    sub_dir_counter = 0

    persons = []

    person_counter = 0
    for line in tsv_file:
        #parse record
        fields = line.split("\t")
        some_id = fields[0]
        name = fields[1]
        num = fields[2]
        url_0 = fields[3]
        url_1 = fields[4]
        # print fields
        base64_image = fields[5]
        # convert to byte image and save to name specific dir
        byte_image = base64_image.decode('base64')


        if not name in persons:
            #create dir
            persons.append(name)
            total_counter+=1
        #write file


        name_dir = args.output_dir + "/" + name
        if not name_dir in name_dirs:
            mkpath(name_dir)
            name_dirs.append(name_dir)
        with open(name_dir + "/" + num + ".jpg", "wb") as f:
            f.write(byte_image)
        # save info to separate file
        # counter+=1
        # if counter%1000 ==0:
        #     sub_dir_index+=1
        #     info_file.flush()


def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('--msceleb_zip', type=str, help='MSCeleb1M zip archive', required=True)
    parser.add_argument('--output-dir', type=str, help='Directory to store extracted images grouped by person name', required=True)
    parser.add_argument('--max-dirs', type=int, help='Num of dirs in one dir', default=5000)
    return parser.parse_args(argv)


if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))