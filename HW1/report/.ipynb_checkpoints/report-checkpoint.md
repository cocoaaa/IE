# Assignment 1 report: NER tagging
- Hayley Song (haejinso@usc.edu)

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


# Part 1: CRF
For the implementation of Conditional Random Field(CRF), I used the [sklearn_crf]() library. 
I got a good amount of ideas from this [tutorial](). The implementation of CRF is based on this [paper]().
I included detailed steps and explanations with corresponding codes for this part of the assigmenent in
`notebooks/ner_crf.ipynb`. Please read that file.

Instead of repeating the same details, here I summarize the main components for my CRF model.
First, the features I used for this vanilla CRF classifier are the following:

    word in lowercase, POS, suffix2, suffix3, word shape info, prev_word in lowercase, prev_word_POS, prev_word_shape, next_word in lowercase, next_word_POS, next_word_shape,
    location in the sentence (BOS, EOS)
    
Features like POS (Part of Speech), suffix2 and suffix3 are unique to the word, and we add features from neighboring
words in order to incorporate contexts,

For instance, the word "Germany" is represented as the following vector by using these features:
    
    {'word': 'germany', 'pos': 'NNP', 'pos2': 'NN', 'last2': 'ny', 'last3': 'any', 'isUpper': False, 'isTitle': True, 'isAlpha': True, 'isDigit': False, 
    'next_word': "'s", 'next_pos': 'POS', 'next_pos2': 'PO', 'next_last2': "'s", 'next_last3': "'s", 'next_isUpper': False, 'next_isTitle': False, 'next_isalpha': False, 'next_isdigit': False, 
    'bos': True}


1. Initial model with randomly chosen hyperparameters
Table 1
![Table 1](images/crf1_weights.png)

Table 1 shows how each of these feature contribute the model's performances.
After training the model with randomly chosen hyperparameters (c1, c2), I inspected the learned weights.
Many of the features with high values of weights were, unfortunately, specific to particular words. For instance,


2. Cross-Validated model
Table 2 shows the performances of the best model from the cross-validation on `testa`, and Table 3
shows the learned weights and the label to label transition
probabilities.  

Table 2
![Table 2](images/best_crf_performances.png) 

Table 3 
![Table 3](images/best_crf_weights.png)  

However, we still see that the model "remembered" word-specific features such as (the word itself like "Germany")
rather than giving a high weight for more generalizable features like POS and suffix.  

3. Regularized model
Even the best performing model from the cross-validation still suffers from giving high weights to word-spefici features.
In order to alleviate this, I tried regularization. Table 4 shows the regularized model's performaces on the training set, and 
Table 5 shows the performances on the dev set (ie. `testa`). Table 6 shows the learned weights and transition probabilities.

Table 4 
![Table 4](images/crf_reg_train_performances.png) 

Table 5 
![Table5](images/crf_reg_dev_performances.png)  

Table 6 
![Table 6](images/crf_reg_weights.png)  



## Summary
I used both word-specific features (eg. the word itself, its part-of-speech, suffix and word shapes) as well
as the features from its direct neighbors to train the CRF model.  Using the cross-validation, I searched the 
hyperparameter spaces and tuned the model.  The best CV settings were used to train the model on the entire 
training set.  As a result, the best CV model achieved 0.9 precision, 0.874 recall and 0.887 f1-score on the
provided test set.  Its learned weights and transition probabilities are showned in Table 3. In order to alleviate
the overfitting observed in our cross-validated model, I retrained the model with regularization.  The results on
train and dev sets are shown in Table 4 and 5, and the learned parameters are shown in Table 6.  However, I would
like to note that the scores from training set and dev set suggests the model is still overfitted. 

I use this regularized model to make a prediction on `testb`.  The predicted tags are saved as 
`report/predictions/crf.bio` for the BIO scheme and `report/predictions/crf.ibo` for the IBO scheme.  The scheme conversion 
was done using the code provided by the TA for this assignment. 

# Part 2: Bidirectional LSTM + CRF
- modification
- explain the architecture
- add a figure (draw and take a picture)
- learning rate?

- viterbi algorithm
- hyperparameters



