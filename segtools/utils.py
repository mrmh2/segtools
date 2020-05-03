import numpy as np
import pandas as pd

from dtoolbioimage import Image


def cell_centroid_csv_to_cell_centers_dict(cell_centers_csv_fpath):

   cell_centers_df = pd.read_csv(cell_centers_csv_fpath)
   cell_centers = {p.Index: (p.X, p.Y) for p in cell_centers_df.itertuples()}

   return cell_centers


def create_overlaid_image(projection, segmentation):
    merged = 0.5 * segmentation.pretty_color_image + \
        0.5 * np.dstack(3 * [projection])
    return merged.view(Image)
