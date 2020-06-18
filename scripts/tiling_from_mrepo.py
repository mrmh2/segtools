import math
import pathlib
from itertools import zip_longest
import click
import parse
import numpy as np
import skimage.transform
from dtoolbioimage import Image as dbiImage

from PIL import Image, ImageDraw, ImageFont

from mrepo import ManagedRepo


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return zip_longest(*args, fillvalue=fillvalue)


def imdict_and_ordering_to_2D_tiled_composite(imdict, ordering):
    """Given a dictionary mapping keys to images, and an ordering specified as
    a list of lists of keys, generate a tiled composite of the images with the
    given ordering.
    """

    images = [
        [imdict[n] for n in row]
        for row in ordering
    ]

    composite = np.vstack(
        [np.hstack(row) for row in images]
    )

    return composite.view(dbiImage)


def dirpath_to_imdict(dirpath, format_str):
    diter = pathlib.Path(dirpath).iterdir()

    def fpath_to_sid_and_image(fpath):
        result = parse.parse(format_str, str(fpath))
        sid = result.named['key']
        return sid, dbiImage.from_file(fpath)

    imdict = dict(fpath_to_sid_and_image(fpath) for fpath in diter)

    return imdict


def generate_labelled_imdict(imdict):

    labelled_imdict = {}
    for idx, imarray in imdict.items():
        pilim = Image.fromarray(imarray)
        draw = ImageDraw.ImageDraw(pilim)
        fnt_size = 72
        fnt = ImageFont.truetype('Microsoft Sans Serif.ttf', fnt_size)
        draw.text((50, 50), str(idx), font=fnt, fill=255)
        labelled_imdict[idx] = np.array(pilim)

    return labelled_imdict


@click.command()
@click.argument('base_dirpath')
def main(base_dirpath):

    mr = ManagedRepo(base_dirpath)
    
    all_specs = mr.item_specs_by_dataspec()['projection']
    selected_specs = [item for item in all_specs if item.tp == 304]

    dataspec = {
        "type_name": "projection",
        "ext": "png"
    }

    imdict = {
        item.series_index: dbiImage.from_file(mr.item_abspath(item, dataspec))
        for item in selected_specs
    }

    # ordering_244 = [
    #     [5, 2, 0],
    #     [1, 6, 7],
    #     [4, 3, 8]
    # ]

    n = len(imdict)
    ordering = grouper(iter(range(n)), math.floor(math.sqrt(n)))
    # ordering_256 = [
    #     [5, 7, 2],
    #     [8, 6, 0],
    #     [3, 4, 1]
    # ]

    full_ordering_280 = [
        [0, 2, 3, 13],
        [9, 4, 15, 10],
        [14, 12, 5, 8],
        [6, 7, 1, 11]
    ]

    ordering_280 = [
        [12, 5, 8],
        [7, 1, 11]
    ]

    ordering = [
        [3, 15, 21, 19, 2],
        [8, 14, 1, 13, 20],
        [23, 7, 11, 5, 18],
        [23, 23, 22, 6, 12],
        [23, 23, 0, 9, 4]
    ]

    labelled_imdict = generate_labelled_imdict(imdict)
    composite = imdict_and_ordering_to_2D_tiled_composite(labelled_imdict, ordering)
    composite.save("composite.png")


if __name__ == "__main__":
    main()
