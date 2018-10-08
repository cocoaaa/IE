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

- Build models
I tried three different models for this assignment: Vanilla CRF, LSTM, and bidirectional LSTM with CRF.
Details will follow in Part 1 and 2 of this report.

- Train and model tuning/selection

- Make a prediction on `testb`


# Part 1: Vanilla CRF
For the implementation of Conditional Random Field(CRF), I used the 
- features
- Point to the notebook for how different features affect the performance
- Add screenshot of the results

# Part 2: Bidirectional LSTM + CRF
- modification
- explain the architecture
- add a figure (draw and take a picture)
- learning rate?

- viterbi algorithm
- hyperparameters



