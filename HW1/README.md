# Overview

This code repository provides the data, utility functions and evaluation script to help you start your project.

## Data

Both train, testa and (unlabeled) testb are provided. It's stored in `data/` folder.

## Utility Functions

We provide the functions to:

- Convert CONLL tagging scheme to schemes we are familar with (bio, bioe and bioes).
- Read and write in CONLL format.

It's stored in `utils/data_converter.py`. The `main()` function illustrates how to use these functions to transform CONLL data.

## Evaluation

The offical evaluation script is written in `perl`, which is hard to integrate with current systems. [Here]() is a re-written version of evalatuion script in python, which I included in `utils/conlleval.py`. 

To run this evaluation script, run command line:

`python conlleval.py ${PREDICTION_FILE}`

Or invoke function `evaluate()` directly on data streams. See `utils/data_converter.py` for examples.
