Segtools
========

Tools to create and work with segmentations, including tiling.

How to tile images
------------------

Setup
~~~~~

Make sure you have installed:

* https://github.com/mrmh2/mrepo
* https://github.com/mrmh2/runtools

and that they are in the Python path:

.. code-block:: bash

    export PYTHONPATH=~/projects/mylibs/mrepo:~/projects/runtools/

Then you will need to create a YAML file to guide the tiling. There is an
example in the `example_input` folder. The contents of the YAML file will
look like this:

.. code-block:: YAML

    mrepo_dirpath: /Users/mhartley/workingdata/da1-mrepo
    position: FR
    genotype: WT
    tp: 252
    ordering: [[1, 2], [0, 3]]

mrepo_dirpath should point at the working data directory. Ordering is the order
of the tiles.

Then run:

.. code-block:: bash

    python scripts/autotile_from_yaml.py tile_wt_fr_288.yml --guide

This will output test .png files which you can use to check the ordering. Rerrange
the ordering list, then run

.. code-block:: bash

    python scripts/autotile_from_yaml.py tile_wt_fr_288.yml

ordering is a list, 

This creates the -finalprojection.png and -final-segmentation.png files
