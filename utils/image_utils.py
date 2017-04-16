from sklearn.feature_extraction.image import extract_patches_2d, reconstruct_from_patches_2d


def get_image_patches(image, crop_size=[40, 40], stride=20):
    shape = image.shape
    patches = []
    for x in range(0, shape[0], stride):
        if x + crop_size[0] < shape[0]:
            for y in range(0, shape[1], stride):
                if y + crop_size[1] < shape[1]:
                    patches.append(image[x:x + crop_size[0], y:y + crop_size[1]])

    return patches


def sliding_2d_filtration(image, patch_size=40, filter=None):
    patches = extract_patches_2d(image, [patch_size, patch_size])
    f_patches = []
    patch_count = len(patches)
    if filter is not None:
        for i in range(0, patch_count):
            filtered_patch = filter(patches[i])
            f_patches.append(filtered_patch)

    f_image = reconstruct_from_patches_2d(f_patches, image.shape)

    return f_image
