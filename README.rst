Segtools
========

Tools to create and work with segmentations, including tiling.

How to tile images
------------------

mrepo_dirpath: /Users/mhartley/workingdata/da1-mrepo
position: FR
genotype: WT
tp: 252
ordering: [[1, 2], [0, 3]]

mrepo_dirpath should point at the working data directory.

ordering is a list, 

.. code-block:: bash

    export PYTHONPATH=~/projects/mylibs/mrepo:~/projects/runtools/
    python scripts/autotile_from_yaml.py tile_wt_fr_288.yml --guide


Then, to finalise, run:


.. code-block:: bash

    python scripts/autotile_from_yaml.py tile_wt_fr_288.yml

This creates the -finalprojection.png and -final-segmentation.png files
