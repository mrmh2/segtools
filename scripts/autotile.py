import logging

import numpy as np
from skimage.feature import match_template

import click
from dtoolbioimage import Image as dbiImage
from dtoolbioimage.segment import Segmentation

from mrepo import ManagedRepo, filter_specs


logger = logging.getLogger(__file__)


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


def segjoin(mr, id_from, id_to):

    all_specs = mr.item_specs_by_dataspec()['segmentation']
    selected_specs = [item for item in all_specs if item.tp == 256]

    segmentation_spec = {
        "type_name": "segmentation",
        "ext": "png"
    }
    segmentations = {
        item.series_index: Segmentation.from_file(
            mr.item_abspath(item, segmentation_spec))
        for item in selected_specs
    }

    left = np.array(segmentations[id_from].label_id_image)[:, 768:, :]
    right = np.array(segmentations[id_to].label_id_image)[:, :256, :]
    np.hstack([left, right]).view(dbiImage).save(
        f"joinimage-{id_from}-{id_to}.png")

    joins_4_1 = {
        72: 17,
        941: 381,
        288: 385,
        501: 182,
        585: 223,
        746: 308,
        840: 357
    }

    joins_8_6 = {
        27: 585,
        281: 593,
        47: 78,
        89: 173,
        136: 269,
        182: 390,
        212: 505
    }

    joins_6_0 = {
        354: 64,
        245: 44,
        93: 17,
        60: 11
    }

    joins = {
        (4, 1): joins_4_1,
        (8, 6): joins_8_6,
        (6, 0): joins_6_0
    }

    joins_3_4 = {
        7: 73,
        12: 101,
        2: 45,
        17: 154
    }


    print(delta_from_segs_and_joins(
        segmentations[id_from],
        segmentations[id_to],
        joins[(id_from, id_to)])
    )


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


@click.command()
@click.argument('base_dirpath')
def oldmain(base_dirpath):

    tp = 256

    mr = ManagedRepo(base_dirpath)

    all_specs = mr.item_specs_by_dataspec()['projection']
    selected_specs = [item for item in all_specs if item.tp == tp]
    projection_spec = {
        "type_name": "projection",
        "ext": "png"
    }
    projections = {
        item.series_index: dbiImage.from_file(mr.item_abspath(item, projection_spec))
        for item in selected_specs
    }

    all_specs = mr.item_specs_by_dataspec()['segmentation']
    selected_specs = [item for item in all_specs if item.tp == tp]
    segmentation_spec = {
        "type_name": "segmentation",
        "ext": "png"
    }
    segmentations = {
        item.series_index: Segmentation.from_file(
            mr.item_abspath(item, segmentation_spec))
        for item in selected_specs
    }

    ordering_256 = [
        [8, 6, 0],
        [3, 4, 1]
    ]

    ordering_280 = [
        [12, 5, 8],
        [7, 1, 11]
    ]

    ordering_304 = [
        [22, 6, 12],
        [0, 9, 4]
    ]

    ordering = ordering_256
    row = ordering[0]

    tdr, tdc = 0, 0
    offsets = {
        row[0]: (0, 0)
    }
    for i in range(len(row)-1):
        id_from = row[i]
        id_to = row[i+1]
        print(f"Join {id_from} to {id_to}")
        dr, dc = delta_join_by_matching(projections[id_from], projections[id_to])
        tdr += int(dr)
        tdc += int(dc)
        print(id_to, tdr, tdc)
        offsets[id_to] = tdr, tdc


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
        print(f"Join {id_from} to {id_to}")
        dr, dc = delta_join_by_matching(projections[id_from], projections[id_to])
        tdr += int(dr)
        tdc += int(dc)
        print(id_to, tdr, tdc)
        offsets[id_to] = tdr, tdc

    adjusted_offsets = adjust_offsets(offsets)
    merge_and_save_projections(projections, adjusted_offsets, "pmerge.png")

    segs_offsets = [
        (segmentations[idx], offset)
        for idx, offset in adjusted_offsets.items()
    ]
    merged_segmentation = segmentations_and_offsets_to_merged_segmentation(segs_offsets)
    merged_segmentation.save("smerge.png")


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

    if len(ordering) == 2:
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

    return offsets


@click.command()
@click.argument('base_dirpath')
@click.argument('tp', type=float)
def main(base_dirpath, tp):

    mr = ManagedRepo(base_dirpath)
    all_specs = mr.item_specs_by_dataspec()['projection']

    position = "BR"
    genotype = "WT"

    # ordering = [[0, 1]]
    ordering = [[2, 1], [0, 3]]

    selected_specs = filter_specs(all_specs, tp=tp, genotype=genotype, position=position)
    projections = load_projections(mr, selected_specs)
    segmentations = load_segmentations(mr, selected_specs)

    offsets = determine_offsets_from_projections(projections, ordering)

    adjusted_offsets = adjust_offsets(offsets)

    merge_and_save_projections(projections, adjusted_offsets, "pmerge.png")
    merge_and_save_segmentations(segmentations, adjusted_offsets, "smerge.png")

if __name__ == "__main__":
    main()
