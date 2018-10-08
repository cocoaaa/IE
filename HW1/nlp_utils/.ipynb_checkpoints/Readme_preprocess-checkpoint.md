# Folder description
1. create_vocab.ipynb: contains functions for processing labelled
and unlabelled datasets:
- process_labelled_dataset
- process_unlabelled_dataset
Use these functions to create train, dev, test dataset.
At the end the processed [sentences, labels] list is stored to
a data_dir.

It also creates the the following lookup tables:
- word2idx (dict or json obj)
- tag2idx (dict or json obj)
- idx2word (dict or json obj)
At the end the processed dictionaries are stored in the `data_dir`.

Please refer to each function's documentation for details.