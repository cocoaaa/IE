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
Table 5 shows the performances on the dev set (ie. `testa`). Table 6 and 7 shows the learned weights and transition probabilities.
Comparing this result with the result from the initial, randomly guessed hyperparameter setting of c1=c2=0.1, we see a decent improvement. 
Let's further inspect what the classifier learned. In order to do so, we inspect the learned transition (state to state, i.e label to label)
probability distribution, and the probability of output labels.

<figure>
    <img src= 'images/crf_reg_train_performances.png' />
    <figcaption> Table 4: regularized CRF model performances on train data</figcaption>    
</figure>

<figure>
    <img src= 'images/crf_reg_dev_performances.png' />
    <figcaption> Table 5: regularized CRF model performances on dev data</figcaption>    
</figure>

Let's first take a look at the learned transition matrix. The transition probabilities look reasonable as it shows high probabilities of 
transitioning from `B-TYPE` to `I-TYPE`. For instance, `I-LOC` to `I-LOC`, `B-MISC` to `I-MISC` and `B-ORG` to `I-ORG` have highest weights.
Conversely, the model learned that it's unlikely to see a beginning tag of a type to be followed by an intermediate tag of a different type:
Pr{`B_ORG` -> `t`} are either very close to 0 or negative for any `t` that is not `I-ORG`. `B-PER` shows the same trend. 

<figure>
    <img src= 'images/best_transition.png' />
    <figcaption> Table 6: regularized CRF model's transition probabilities</figcaption>    
</figure>

Now let's examine the weights the model learned from training.

<figure>
    <img src= 'images/best_weights.png' />
    <figcaption> Table 7: regularized CRF model's learned weights</figcaption>    
</figure>

First of all, notice how setting the parameter $c_1$ for the $l_1$ regularizer drove many weights to be close to zero.  
We can see that the regualization is doing the right job of making the weight parameters sparse. To see if this high regualization 
hyperparameter for $l_1$ term, we further inspect the transition probabilities and the learned weights. Let's focus on the second
columns of Table 4 with Table 3. We discussed earlier that the unregularized model memorized specific words word-by-word and put 
large weights on them to compute the output probabilities. Many of the important features were exact instances encountered in the
training dataset, such as "chester-le-street", "gemerny" and "hungary". However, this overfitting is alleviated through increasing 
the $l_1$ regualization. The regualized model assigns high weights to more generalizable features such as "bos" 
(i.e. beginning of the sentence), `prev_las2:AT` and `last2:ia`. We observe this improvement in other classes as well.  
Column for `Y=I-ORG` shows high weights for features like `prev_word:Cooperation`, `word:inc`, `last3:oom`, `next_isDigit` and 
`next_last3:ker` which are all intuitively reasonable (for instance a lot of companies name starts with `Cooporation...`) and not 
specific to a single training instance, like "Microsoft". For `I-PER` class, top most important features were `next_pos:NNPS`, 
`last2:ez`, and `isTitle`.  When I experimented with even larger $l_1$ norm ($c_1$ = 10, $c_2=0.05$), I observed even more sparsity 
in the weight matrix. In other words, the model selected smaller set of features to be meaningful for the classification. 
The features also seem to become more generalizable. For instance, for the `I-PER` class, top important features were `last2:er`, 
`last2:on`, `prev_isUpper` and `pos:NNP`.   These features surprisingly well match how a human would decide whether an entity is of 
`PERSON`.  

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
stored in `log/progress_10_01`.  In this section, I will summarize my data preprocessing and loading processes 
as well as the training and evaluation processes. 
### a. Data Processing
Unlike the CRF models, we don't need to hand-engineer the features for our word representation. 
We use the index mapper (from word to index and tag to index) as we defined in `create_vocab.ipynb` and in the `Step` section above.
To summarize, we use the most common $N$ words out of the total words that occured in all the datasets provided 
(`eng:train`, `eng:testa`, `eng:testb`). In our experiments $N$ is set $15,000$ which is half of the number of words 
that occured at least once in any of the given datasets. I added several extra words: PAD_WORD and UNK_WORD. PAD_WORDs 
are appended to sentences in a mini-batch to make every sentence of equal length (as to the length of the longest sentence 
in the mini-batch). UNK_WORD is used to map words that are not in our vocab (because we excluded 15,000 uncommon words). 
In addition, I assigned a PAD_TAG to indicate the padding words, and START_TAG and STOP_TAG. 

<figure>
    <img src= 'images/pad_and_unk.png' height="400"/>
</figure>

### b. Data Loader
When we sample a batch of sentences, the sentences usually have a different length. In a batch of sentences, 
(`batch_sentences`) with correspoonding batch of tags `batch_tags`, we add PAD_WORD for sentences that have fewer words 
than SQE_LENGTH (set to maximum length of a sentence in the current `batch_sentences`). Below shows this processing procedure.

```python
# This is just to show the processing and is not meant to actually run.
# Maximum sentence lengths in current batch 
batch_max_len = max([len(s) for s in batch_sentences])

# Intial matrix
batch_data = word2idx[PAD_WORD]*np.ones((len(batch_sentences), batch_max_len))
batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))

# Fill in the matrix with current batch sentences and labels
for i in range(len(batch_sentences)):
    curr_len = len(batch_sentences[i])
    batch_data[i][:curr_len] = batch_sentences[i]
    batch_labels[i][:curr_len] = batch_tags[i]

# Convert to torch.LongTensors (since each entry is an index)
batch_data = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)
               
```

### c. Model Architecture
This model is composed of three components.
- an embedding layer that maps each index in range(vocab_size) to a embedding_dim vector
- LSTM that takes a sequence (sentence) of words and returns an output of tag for each token in the input.
- Fully-connected layer that takes in the output of the LSTM unit and converts it to a distribution 
    over the set of NER tags
    
The size of the embedding and the LSTM unit's output dimension are parameters of this model.
Other parameters I experimented with are `batch_size`, `learning_rate`, and `number of epochs`.
These settings are stored in `data/{params_name}.json`. For my base model, I tried the following setting:

<figure>
    <img src= 'images/base_params.png' height="200" width="200"/>
</figure>

I used the following setting for my cross-validation:
<figure>
    <img src= 'images/params1.png' height="200" width="200"/>
</figure>

I used Adam as my optimizer. 

### d. Evaulation 
For evaluating different models, I used the entity-level average F1 score. This score is calculated by excluding 
the padded words so that the padding procedure at each mini-batch doesn't affect model evaluations. 

## Model 2: Bidirectional LSTM + CRF
# 2-2. NER with LSTM + CRF
Another experiement I ran for the NER task with RNN based models is a bidirectional LSTM + CRF. I trained a Bidirectional LSTM
to learn the word embeddings and input to output mapping, and passed the output word vectors as input sequences to the CRF
to learn the output probability over the tag space. Please refer to `notebook/ner_lstm_crf.ipynb` for this model's implementation 
details. 

### a. Model architecture
Please refer to the `BILASTM_CRF` class in `notebook/bilstm_crf.py` for implementation details.
This model has the following components:
- embedding: word embedding layer
- Bidirectional LSTM unit
- CRF 
    - Single FC layer that maps the output of BiLSTM to the tag probabilities
    - Transition matrix for CRF that indicates the probability of transitioning from tag $j$ to tag $i$
   
The CRF internally implements Viterbi algorithm for forward and backward passing to update the transition matrix.
I used the cross_entropy as the loss function and Adam as the optimizer.

---
For the prediction on `testb` using RNN-based models, I used the first model (LSTM+FC).
The prediction procedure uses `model_evaluation.py` in the `nlp_utils` folder.