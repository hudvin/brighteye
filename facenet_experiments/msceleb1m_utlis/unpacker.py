import zipfile
from distutils.dir_util import mkpath

out_root = "/media/3tb/media/MsCeleb1MV1"
zip_archive_path = "/media/3tb/media/MsCelebV1-ImageThumbnails.zip"

zip_archive = zipfile.ZipFile(zip_archive_path)
tsv_file = zip_archive.open('MsCelebV1-ImageThumbnails.tsv', "r")

info_file_path = out_root + "/" + "info.csv"
info_file = open(info_file_path, "w")
info_file.write(",".join(["some_id", "name", "num", "url_0", "url_1"]) + "\n")

name_dirs = []

# counter = 0
sub_dir_index = 0
sub_dir_counter = 0
for line in tsv_file:
    fields = line.split("\t")
    some_id = fields[0]
    name = fields[1]
    num = fields[2]
    url_0 = fields[3]
    url_1 = fields[4]
    #print fields
    base64_image = fields[5]
    #convert to byte image and save to name specific dir
    byte_image = base64_image.decode('base64')
    name_dir = out_root+ "/"+ str(sub_dir_index) + "/"+ name
    if not name_dir in name_dirs:
        sub_dir_counter+=1
        if sub_dir_counter%1000 == 0:
            print sub_dir_index
            sub_dir_index+=1
            info_file.flush()
        mkpath(name_dir)
        name_dirs.append(name_dir)
    with open(name_dir+"/"+num+".jpg", "wb") as f:
        f.write(byte_image)
    #save info to separate file
    info_file.write(",".join([some_id, name, num, url_0, url_1]) + "\n")
    # counter+=1
    # if counter%1000 ==0:
    #     sub_dir_index+=1
    #     info_file.flush()

info_file.close()