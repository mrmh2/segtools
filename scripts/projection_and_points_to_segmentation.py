import click

from dtoolbioimage import Image as dbiImage, scale_to_uint8

from segtools.utils import cell_centroid_csv_to_cell_centers_dict
from segtools.segmentation import image_and_seed_dict_to_ws_segmentation


@click.command()
@click.argument('projection_fpath')
@click.argument('seed_csv_fpath')
@click.argument('output_fpath')
def main(projection_fpath, seed_csv_fpath, output_fpath):

    projection = dbiImage.from_file(projection_fpath)
    seed_dict = cell_centroid_csv_to_cell_centers_dict(seed_csv_fpath)

    segmentation = image_and_seed_dict_to_ws_segmentation(projection, seed_dict)

    segmentation.save(output_fpath)


if __name__ == "__main__":
    main()
