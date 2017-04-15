def get_image_patches(image, crop_size=[40, 40], stride=20):
    shape = image.shape
    patches = []
    for x in range(0, shape[0], stride):
        if x + crop_size[0] < shape[0]:
            for y in range(0, shape[1], stride):
                if y + crop_size[1] < shape[1]:
                    patches.append(image[x:x + crop_size[0], y:y + crop_size[1]])

    return patches
