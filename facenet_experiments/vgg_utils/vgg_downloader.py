import Image
import argparse
from StringIO import StringIO
from urlparse import urlparse
from threading import Thread
import httplib, sys
from Queue import Queue
import numpy as np
from scipy import misc
import os

def doWork():

    while True:
        task_data = q.get()
        print task_data
        url = task_data["url"]
        image_path = task_data["image_path"]
        error_path = task_data["error_path"]
        try:
            url = urlparse(url)
            conn = httplib.HTTPConnection(url.netloc)
            conn.request("GET", url.path)
            res = conn.getresponse()
            if res.status == 200:
                img = res.read()
                img = np.array(Image.open(StringIO(img)))
                misc.imsave(image_path, img)
            else:
                save_error(error_path, res.status + " " + res.reason)
        except Exception as e:
            save_error(error_path, str(e))
        q.task_done()


def save_error(error_path, error_message):
    with open(error_path, "w") as textfile:
        textfile.write(error_message)


concurrent = 200
q = Queue(concurrent * 2)


def main(args):
    for i in range(concurrent):
        t = Thread(target=doWork)
        t.daemon = True
        t.start()
    try:
        textfile_names = os.listdir(args.dataset_descriptor)
        for textfile_name in textfile_names:
            if textfile_name.endswith('.txt'):
                with open(os.path.join(args.dataset_descriptor, textfile_name), 'rt') as f:
                    lines = f.readlines()
                dir_name = textfile_name.split('.')[0]
                class_path = os.path.join(args.output_dir, dir_name)
                if not os.path.exists(class_path):
                    os.makedirs(class_path)
                for line in lines:
                    x = line.split(' ')
                    filename = x[0]
                    url = x[1]
                    image_path = os.path.join(args.output_dir, dir_name, filename + '.' + args.output_format)
                    error_path = os.path.join(args.output_dir, dir_name, filename + '.err')
                    q.put({
                        "url": url.strip(),
                        "image_path":image_path,
                        "error_path":error_path
                    })
        q.join()
    except KeyboardInterrupt:
        sys.exit(1)

def parse_arguments(argv):
    parser = argparse.ArgumentParser()
    parser.add_argument('dataset_descriptor', type=str,
        help='Directory containing the text files with the image URLs. Image files will also be placed in this directory.')
    parser.add_argument('output_dir', type=str,
                        help='Directory to store fetched images grouped by person name')
    parser.add_argument('--output_format', type=str, help='Format of the output images', default='png', choices=['png', 'jpg'])

    return parser.parse_args(argv)

if __name__ == '__main__':
    main(parse_arguments(sys.argv[1:]))
