from os import walk

import cv2


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


def load_image_filename_list(filename_list, gray=False):
    image_list = []
    for filename in filename_list:
        image = cv2.imread(filename)
        if gray:
            image = cv2.cvtColor(image, cv2.COLOR_RGB2GRAY)
        image_list.append(image)

    return image_list
