import logging
import itertools

import numpy as np
from skimage.feature import match_template

import click
import parse
from dtoolbioimage import Image as dbiImage
from dtoolbioimage.segment import Segmentation

from mrepo import ManagedRepo, filter_specs

from runtools.config import Config


logger = logging.getLogger(__file__)


def grouper(iterable, n, fillvalue=None):
    "Collect data into fixed-length chunks or blocks"
    # grouper('ABCDEFG', 3, 'x') --> ABC DEF Gxx"
    args = [iter(iterable)] * n
    return itertools.zip_longest(*args, fillvalue=fillvalue)


def delta_join_by_matching(proj_from, proj_to, mode='right'):
    ts = 50

    rows = np.linspace(50, 950, 11).astype(int)

    if mode == 'right':
        axis_vector = np.array([1, 0])
        offset_vector = np.array([0, 974])

    if mode == 'down':
        axis_vector = np.array([0, 1])
        offset_vector = np.array([974, 0])

    tile_positions = [
        k * axis_vector + offset_vector
        for k in np.linspace(50, 950, 21).astype(int)
    ]

    templates = [
        proj_from[r:r+ts, c:c+ts]
        for r, c in tile_positions
    ]

    match_results = [
        match_template(proj_to, template, pad_input=True) 
        for template in templates
    ]

    tlocs = [
        np.unravel_index(match_result.argmax(), proj_to.shape)
        for match_result in match_results
    ]

    match_scores = [mr.max() for mr in match_results]
    indices = np.where(np.array(match_scores) > 0.5)[0]

    all_deltas = [np.array((ts//2, ts//2)) - tl + tp for tl, tp in zip(tlocs, tile_positions)]

    # print(match_scores)

    return np.median(np.array(all_deltas)[indices], axis=0)


def delta_from_segs_and_joins(seg_from, seg_to, joins):
    deltas = np.array([
        np.array(seg_from.rprops[lfrom].centroid) -
        np.array(seg_to.rprops[lto].centroid)
        for lfrom, lto in joins.items()
    ])

    print(deltas)

    return np.median(deltas, axis=0)


def projections_and_offsets_to_merged_projection(projs_offsets, tdim=1024):

    max_dr = max([dr for _, (dr, _) in projs_offsets])
    max_dc = max([dc for _, (_, dc) in projs_offsets])
    canvas = np.zeros((1024+max_dr, 1024+max_dc), dtype=np.uint16)

    print(max_dr, max_dc)

    for proj, (dr, dc) in projs_offsets:
        print(dr, dc)
        canvas[dr:dr+tdim, dc:dc+tdim] = proj

    return canvas


def merge_and_save_projections(projections, offsets, output_fpath):
    projs_offsets = [
        (projections[idx], offset)
        for idx, offset in offsets.items()
    ]

    merged_projection = projections_and_offsets_to_merged_projection(
        projs_offsets).view(dbiImage)

    merged_projection.save(output_fpath)


def merge_and_save_segmentations(segmentations, offsets, output_fpath):
    segs_offsets = [
        (segmentations[idx], offset)
        for idx, offset in offsets.items()
    ]

    merged_segmentation = segmentations_and_offsets_to_merged_segmentation(segs_offsets)

    merged_segmentation.save(output_fpath)


def adjust_offsets(offsets):

    rdeltas, cdeltas = zip(*offsets.values())
    roffset = -min(rdeltas)
    coffset = -min(cdeltas)

    adjusted_offsets = {
        idx: (r+roffset, c+coffset)
        for idx, (r, c) in offsets.items()
    }

    return adjusted_offsets


def append_segmentation(canvas, segadd, delta, label_iter):
    mask_points = {tuple(p) for p in np.vstack(canvas.nonzero()).T}
    dr, dc = delta
    for l in segadd.labels:
        rr, cc = np.where(segadd == l)
        rr += dr
        cc += dc
        cell_points = set(zip(*(rr, cc)))
        overlap_fraction = len(cell_points & mask_points) / len(cell_points)
        if overlap_fraction < 0.5:
            canvas[(rr, cc)] = next(label_iter)


def segmentations_and_offsets_to_merged_segmentation(segs_offsets, tdim=1024):
    # nrows = 2
    # ncols = 3
    # canvas = np.zeros((nrows*tdim, ncols*tdim), dtype=np.uint16)

    max_dr = max([dr for _, (dr, _) in segs_offsets])
    max_dc = max([dc for _, (_, dc) in segs_offsets])
    canvas = np.zeros((1024+max_dr, 1024+max_dc), dtype=np.uint16)

    # print(max_dr, max_dc)
    label_iter = iter(range(1, 10000))

    for seg, (dr, dc) in segs_offsets:
        append_segmentation(canvas, seg, (dr, dc), label_iter)

    return canvas.view(Segmentation)


def load_projections(mr, selected_specs):
    projection_spec = {
        "type_name": "projection",
        "ext": "png"
    }
    projections = {
        item.series_index: dbiImage.from_file(
            mr.item_abspath(item, projection_spec)
        )
        for item in selected_specs
    }

    return projections


def load_segmentations(mr, selected_specs):
    segmentation_spec = {
        "type_name": "segmentation",
        "ext": "png"
    }
    segmentations = {
        item.series_index: Segmentation.from_file(
            mr.item_abspath(item, segmentation_spec))
        for item in selected_specs
    }

    return segmentations


def determine_offsets_from_projections(projections, ordering):

    print(f"Ordering: {ordering}")

    row = ordering[0]

    tdr, tdc = 0, 0
    offsets = {
        row[0]: (0, 0)
    }

    for i in range(len(row)-1):
        id_from = row[i]
        id_to = row[i+1]
        logging.info(f"Join {id_from} to {id_to}")
        dr, dc = delta_join_by_matching(
            projections[id_from], projections[id_to]
        )
        tdr += int(dr)
        tdc += int(dc)
        # print(id_to, tdr, tdc)
        offsets[id_to] = tdr, tdc

    if len(ordering) >= 2:

        logger.info(f"At least 2 rows in ordering")

        row = ordering[1]

        dr, dc = delta_join_by_matching(
            projections[ordering[0][0]],
            projections[ordering[1][0]],
            'down'
        )
        tdr = int(dr)
        tdc = int(dc)
        offsets[row[0]] = (tdr, tdc)
        for i in range(len(row)-1):
            id_from = row[i]
            id_to = row[i+1]
            logging.info(f"Join {id_from} to {id_to}")
            dr, dc = delta_join_by_matching(
                projections[id_from], projections[id_to])
            tdr += int(dr)
            tdc += int(dc)
            offsets[id_to] = tdr, tdc

    if len(ordering) == 3:

        logger.info(f"At least 3 rows in ordering")

        tdr0, tdc0 = offsets[row[0]]

        row = ordering[2]

        dr, dc = delta_join_by_matching(
            projections[ordering[1][0]],
            projections[ordering[2][0]],
            'down'
        )
        tdr = int(dr) + tdr0
        tdc = int(dc) + tdc0
        offsets[row[0]] = (tdr, tdc)
        for i in range(len(row)-1):
            id_from = row[i]
            id_to = row[i+1]
            logging.info(f"Join {id_from} to {id_to}")
            dr, dc = delta_join_by_matching(
                projections[id_from], projections[id_to])
            tdr += int(dr)
            tdc += int(dc)
            offsets[id_to] = tdr, tdc

    return offsets


def create_guide_image(config):

    mr = ManagedRepo(config.mrepo_dirpath)
    all_specs = mr.item_specs_by_dataspec()['projection']

    selected_specs = filter_specs(
        all_specs,
        tp=config.tp,
        genotype=config.genotype,
        position=config.position
    )

    projections = load_projections(mr, selected_specs)

    rowjoins = [
        np.hstack([projections[n] for n in row])
        for row in config.ordering
    ]

    np.vstack(rowjoins).view(dbiImage).save('test.png')


def get_projection_fname(mr, selected_specs):

    format_str = mr.fname_format
    spec = list(selected_specs)[0]
    metadata = vars(spec)
    metadata.update({
        "ext": "png",
        "series_index": 0,
        "type_name": "finalprojection"
    })
    projection_fname = format_str.format(**metadata)

    return projection_fname


def create_merges(config):

    mr = ManagedRepo(config.mrepo_dirpath)
    all_specs = mr.item_specs_by_dataspec()['projection']

    selected_specs = filter_specs(
        all_specs,
        tp=config.tp,
        genotype=config.genotype,
        position=config.position
    )

    selected_specs = list(selected_specs)

    projections = load_projections(mr, selected_specs)
    segmentations = load_segmentations(mr, selected_specs)

    offsets = determine_offsets_from_projections(projections, config.ordering)

    adjusted_offsets = adjust_offsets(offsets)

    print(adjusted_offsets)

    spec = list(selected_specs)[0]
    spec.series_index = 0
    fprojection_spec = {
        "type_name": "finalprojection",
        "ext": "png"
    }
    fsegmentation_spec = {
        "type_name": "finalsegmentation",
        "ext": "png"
    }
    projection_fname = mr.fname_for_spec(fprojection_spec, spec)
    segmentation_fname = mr.fname_for_spec(fsegmentation_spec, spec)

    merge_and_save_projections(projections, adjusted_offsets, projection_fname)
    merge_and_save_segmentations(segmentations, adjusted_offsets, segmentation_fname)


@click.command()
@click.argument('config_fpath')
@click.option('--guide/--no-guide', default=False)
def main(config_fpath, guide):

    logging.basicConfig(level=logging.INFO)


    config = Config.from_fpath(config_fpath)

    if guide:
        create_guide_image(config)
    else:
        create_merges(config)




if __name__ == "__main__":
    main()
