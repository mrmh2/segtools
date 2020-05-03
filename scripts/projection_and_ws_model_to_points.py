import numpy as np
import pandas as pd
import click

from dtoolbioimage import Image as dbiImage, scale_to_uint8
from aiutils.unetmodel import TrainedUNet

from skimage.measure import label, regionprops


def imarray_to_filtered_points(imarray):

    label_im = label(imarray > 0.3)
    rprops = regionprops(label_im)
    seed_points = [tuple(map(int, r.centroid)) for r in rprops]

    def point_in_bounds(p, bs=16):
        r, c = p
        return (r > bs) and (r < (1024-bs)) and (c > bs) and (c < (1024-bs))

    filtered_points = [p for p in seed_points if point_in_bounds(p)]

    return filtered_points


def save_points_list_to_csv(output_fpath, points):
    df = pd.DataFrame(points, columns=['X', 'Y'])

    df.to_csv(output_fpath, index=False)


@click.command()
@click.argument('projection_fpath')
@click.argument('ws_model_uri')
@click.argument('output_fpath')
def main(projection_fpath, ws_model_uri, output_fpath):

    projection = dbiImage.from_file(projection_fpath)
    trained_model =TrainedUNet(ws_model_uri)
    ws_mask = trained_model.predict_mask_from_image(projection)
    filtered_points = imarray_to_filtered_points(ws_mask)

    save_points_list_to_csv(output_fpath, filtered_points)


if __name__ == "__main__":
    main()
