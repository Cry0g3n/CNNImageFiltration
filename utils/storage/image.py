from os import walk

from skimage.io import imread


def get_image_filename_list(path):
    if path is None:
        return []
    image_filename_list = []
    for (dirpath, dirnames, filenames) in walk(path):
        for i in range(0, len(filenames)):
            filenames[i] = dirpath + '\\' + filenames[i]
        image_filename_list.extend(filenames)
        break

    return image_filename_list


def get_image_filename_list2(path):
    from os import listdir
    from os.path import isfile, join
    onlyfiles = [f for f in listdir(path) if isfile(join(path, f))]
    return onlyfiles


def load_image_filename_list(filename_list):
    image_list = []
    for filename in filename_list:
        image = imread(filename)
        image_list.append(image)

    return image_list
