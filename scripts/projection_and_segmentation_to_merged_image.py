import click

from dtoolbioimage import Image
from dtoolbioimage.segment import Segmentation


import numpy as np


def create_overlaid_image(projection, segmentation):
    merged = 0.5 * segmentation.pretty_color_image + 0.5 * np.dstack(3 * [projection])
    return merged.view(Image)
    

@click.command()
@click.argument('projection_fpath')
@click.argument('segmentation_fpath')
@click.argument('output_fpath')
def main(projection_fpath, segmentation_fpath, output_fpath):

    projection = Image.from_file(projection_fpath)
    segmentation = Segmentation.from_file(segmentation_fpath)

    merged_image = create_overlaid_image(projection, segmentation)

    merged_image.save(output_fpath)


if __name__ == "__main__":
    main()
