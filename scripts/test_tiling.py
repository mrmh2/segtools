import pathlib
from itertools import zip_longest
import click
import parse
import numpy as np
import skimage.transform
from dtoolbioimage import Image as dbiImage


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


@click.command()
@click.argument('base_dirpath')
def main(base_dirpath):

    format_str = "{prefix}_{key:d}-projection.png"
    imdict = dirpath_to_imdict(base_dirpath, format_str)

    ordering = [
        [5, 2, 0],
        [1, 6, 7],
        [4, 3, 8]
    ]

    composite = imdict_and_ordering_to_2D_tiled_composite(imdict, ordering)
    composite.save("composite.png")


if __name__ == "__main__":
    main()
