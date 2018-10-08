# Steps
Before designing a model for named entity recognition, we need to preprocess our input text files.

- Convert tagging scheme
First we use the provided script to convert the tagging scheme from IBO to BIO.
The converted files are saved in `./data/train.bio, ./data/testa.bio`. Since `testb` doesn't
contain any label, it doesn't need any processing.  

- Prepare train, dev, test dataset for the supervised learning framework
We map each word and tag in `./data/train.bio, ./data/testa.bio` to indices according to the lookup
tables and split the datasets to create [train_sentences, train_labels], [dev_sentences, dev_labels], 
[test_sentences, test_labels]. This process is done in `notebooks/create_vocab.ipynb`. 
In particular, the following function are defined for processing labelled and unlabelled datasets:

    - process_labelled_dataset
    - process_unlabelled_dataset

We use these functions to create train, dev and test datasets. At the end the processed 
[sentences, labels] list is stored to the data directory, `data`.

We also create the the following lookup tables in this notebook:

    - word2idx (dict or json obj)
    - tag2idx (dict or json obj)
    - idx2word (dict or json obj)
    
As a part of the data preprocessing, we add START_WORD, STOP_WORD, PAD_WORD to `word2idx`, and 
PAD_TAG to 'tag2idx`. At the end the processed dictionaries are stored in `data`.

Please refer to each function's documentation for more details.

- Build models
I tried three different models for this assignment: Vanilla CRF, LSTM, and bidirectional LSTM with CRF.
Details will follow in Part 1 and 2 of this report.

- Train and model tuning/selection

- Make a prediction on `testb`


# Part 1: Vanilla CRF
For the implementation of Conditional Random Field(CRF), I used the [sklearn_crf]() library. 
I got a good amount of ideas from this [tutorial](). The implementation of CRF is based on this [paper]().
I included detailed steps and explanations with corresponding codes for this part of the assigmenent in
`notebooks/ner_crf.ipynb`. Please read that file.

Instead of repeating the same details, here I summarize the main components for my CRF model.
First, the features I used for this vanilla CRF classifier are the following:

    > word in lowercase, POS, suffix2, suffix3, word shape info, prev_word in lowercase, prev_word_POS, prev_word_shape, next_word in lowercase, next_word_POS, next_word_shape,
    location in the sentence (BOS, EOS)
    
Features like POS (Part of Speech), suffix2 and suffix3 are unique to the word, and we add features from neighboring
words in order to incorporate contexts,

For instance, the word "Germany" is represented as the following vector by using these features:
    
    > {'word': 'germany', 'pos': 'NNP', 'pos2': 'NN', 'last2': 'ny', 'last3': 'any', 'isUpper': False, 'isTitle': True, 'isAlpha': True, 'isDigit': False, 
    'next_word': "'s", 'next_pos': 'POS', 'next_pos2': 'PO', 'next_last2': "'s", 'next_last3': "'s", 'next_isUpper': False, 'next_isTitle': False, 'next_isalpha': False, 'next_isdigit': False, 
    'bos': True}


- Point to the notebook for how different features affect the performance
[!Table 1]('crf1_weights.png') shows how each of these feature contribute the model's performances.
After training the model with randomly chosen hyperparameters (c1, c2), I inspected the learned weights.
Many of the features with high values of weights were, unfortunately, specific to particular words. For instance,


[!Table 2]('best_crf_performances.png') shows the performances of the best model from the cross-validation on 
`testa`, and [!Table 3]('best_crf_weights.png') shows the learned weights.


- Add two lines of interpretation



# Part 2: Bidirectional LSTM + CRF
- modification
- explain the architecture
- add a figure (draw and take a picture)
- learning rate?

- viterbi algorithm
- hyperparameters



