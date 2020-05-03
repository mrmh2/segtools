import numpy as np
from dtoolbioimage.segment import Segmentation
from skimage.segmentation import watershed, clear_border


def image_and_seed_dict_to_ws_segmentation(image, seed_dict, sz=2):
    """Generate watershed segmentation from image and seed points. The
    function also removes border regions.
    
    Args:
        image (ndarray): Single channel 2D image as numpy array
        seed_dict (dict): Dictionary mapping labels to points
        sz (int): Size of seed which will be generated from each point
        
    Returns:
        Segmentation: The generated segmentation
    """

    label_image = np.zeros_like(image, dtype=np.uint16)

    for l, (r, c) in seed_dict.items():
        label_image[r-sz:r+sz, c-sz:c+sz] = l+1
    watershed_seg = watershed(image, label_image).view(Segmentation)
    noborder = clear_border(watershed_seg).view(Segmentation)

    return noborder
