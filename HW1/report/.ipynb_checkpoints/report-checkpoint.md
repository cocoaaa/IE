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

    word in lowercase, POS, suffix2, suffix3, word shape info, prev_word in lowercase, prev_word_POS, prev_word_shape, 
    next_word in lowercase, next_word_POS, next_word_shape,location in the sentence (BOS, EOS)
    
Features like POS (Part of Speech), suffix2 and suffix3 are unique to the word, and we add features from neighboring
words in order to incorporate contexts,

For instance, the word "Germany" is represented as the following vector by using these features:
    
    {'word': 'germany', 'pos': 'NNP', 'pos2': 'NN', 'last2': 'ny', 'last3': 'any', 'isUpper': False, 'isTitle': True, 
    'isAlpha': True, 'isDigit': False, 'next_word': "'s", 'next_pos': 'POS', 'next_pos2': 'PO', 'next_last2': "'s", 
    'next_last3': "'s", 'next_isUpper': False, 'next_isTitle': False, 'next_isalpha': False, 'next_isdigit': False, 
    'bos': True}


## a. Base model
For my first exploration, I manually set the hyperparameters $c_1$ and $c_2$ to 0.1 and trained on the `eng:train` 
for $1000$ iterations. At the end of the training, the model learned the transition probabilities (from a tag to another)
in Table 1.  I also inspected how each feature contributed to predicting each type of labels.  For instance, in the second 
diagram in Table 1, each column corresponding to a type of label.  The second column, for instance, shows the importances of 
each feature in outputting the label of `B-LOC`.  
<figure >
    <img src= 'images/crf1_weights.png' />
    <figcaption text-align='center'> <b> Table 1: Base CRF model's learned parameters. The first diagram shows the transition probabilities, and the 
        second diagram shows the weights of features for each target class</b> </figcaption>    
</figure>


Many of the features with high values of weights were, unfortunately, specific to particular words. For instance, take a look
at the second column which corresponds to the importance of weights in outputting the `B-LOC` label. Features with high positive
weights were all the exact names of the locations like "hungary", "france" and "ukraine".  Although these words indeed indicates 
a location (and often the beginning of the location indicator), we would like our model to put more emphasis on more generalizable
features like suffix and word shapes. 


## b. Cross-Validated model  
Table 2 shows the performances of the best model from the cross-validation on `testa`, and Table 3
shows the learned weights and the label to label transition
probabilities.  

<figure style="border: 2px dotted gray;">
    <img src= 'images/best_crf_performances.png' />
    <figcaption style="text-align:center;"> Table 2: best CV CRF model</figcaption>    
</figure>

<figure>
    <img src= 'images/best_crf_weights.png' />
    <figcaption> Table 3: best CV CRF model's learned parameters</figcaption>    
</figure>

However, we still see that the model "remembered" word-specific features such as (the word itself like "Germany")
rather than giving a high weight for more generalizable features like POS and suffix.  

## c. Regularized model  
Even the best performing model from the cross-validation still suffers from giving high weights to word-spefici features.
In order to alleviate this, I tried regularization. Table 4 shows the regularized model's performaces on the training set, and 
Table 5 shows the performances on the dev set (ie. `testa`). Table 6 shows the learned weights and transition probabilities.


<figure>
    <img src= 'images/crf_reg_train_performances.png' />
    <figcaption> Table 4: regularized CRF model performances on train data</figcaption>    
</figure>

<figure>
    <img src= 'images/crf_reg_dev_performances.png' />
    <figcaption> Table 5: regularized CRF model performances on dev data</figcaption>    
</figure>

<figure>
    <img src= 'images/crf_reg_weights.png' />
    <figcaption> Table 6: regularized CRF model's learned parameters</figcaption>    
</figure>

First of all, notice how setting the parameter $c_1$ for the $l_1$ regularizer drove many weights to be close to zero.  
We can see that the regualization is doing the right job of making the weight parameters sparse. To see if this high regualization 
hyperparameter for $l_1$ term, we further inspect the transition probabilities and the learned weights. Let's focus on the second
columns of Table 4 with Table 3. We discussed earlier that the unregularized model memorized specific words word-by-word for deciding 
whether to output `B-LOC` or not.  All the features with high weights were given to tokens learned from the training dataset, such as 
"chester-le-street", "gemerny" and "hungary". However, this overfitting is alleviated through increasing the $l_1$ regualization. 
The regualized model assigns high weights to more generalizable features such as "bos" (i.e. beginning of the sentence), "prev_las2:AT"
and "last2:ia". We observe this improvement in generalization for other class weights as well.  Column for `Y=I-ORG` shows high weights 
for features like `word:co`, `last3:oom`, `next_isDigit` and `word:corp` which are all intuitively reasonable (for instance a lot of 
companies name starts with `Cooporation...`) and not specific to a single training instance, like "Microsoft". For `I-PER` class, 
the most significant weights were assigned to features like `last2:er`, `last2:on`, `prev_isUpper` and `pos:NNP`.  
These features in fact match the rules how a human would decide whether an entity is of PERSON.  

Lastly, I set the regularization weights through 5-fold cross-validation, train the best cv model on the entire `eng:train` 
and `eng:testa` dataset. 


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

# Part 2: RNN models
For Part 2, I experimented with two RNN-based models. The first model consists of an embedding, a LSTM unit and a fully-connected (FC)
layer. The second one consists of an embedding layer, a bidirectional LSTM unit and the CRF layer.
## Model 1: LSTM + FC 
Each step of the data processing, data loading, model designing, training and evaluations are included in 
`notebook/ner_rnn.ipynb`. Please refer to the notebook for details. The final model (and several intermediate models) are
stored in `log/trained_part2`.  In this section, I will summarize my data preprocessing and loading processes 
as well as the training and evaluation processes. 
### a. Data Processing
### b. Data Loader
### c. Model Architecture
### d. Evaulation 
ignore padding, do entity-level

### Analysis


preprocessing, padding, how to compute the scores by excluding the padded words' tags (i) 
## Model 2: Bidirectional LSTM + CRF
Please refer to `notebook/ner_lstm_crf.ipynb` for implementation details. The data processing and loading 
processes are same as in the LSTM + FC model. 
### a. Model architecture
### b. Evaulation
### c. Analysis

- modification
- explain the architecture
- add a figure (draw and take a picture)
- learning rate?

- viterbi algorithm
- hyperparameters



