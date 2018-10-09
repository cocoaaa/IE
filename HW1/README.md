# CSC1 699: NER tagging
This folder is for the assignment 1 of CSCI 699 on NER tagging.

- nlp_utils: contains helper functions for data processing, loading as well as 
training and evaluating pytorch models

- notebooks: contains experiments for Part 1 and Part 2. 
    - `create_voab.ipynb`: shows my data preparation process
    - `ner_crf.ipynb`: contains CRF based NER tagger
    - `ner_crf_reg.ipynb`: contains regularied CRF based NER tagger
    - `ner_rnn_1.ipynb`: contains LSTM + FC NER tagger
    - `ner_rnn_2.ipynb`: contain Bidirectional LSTM + CRF tagger
    - `bilstm_crf.py`: contains BILSTM_CRF class definition
    
    - `ner_rnn_1_perp_submission.ipynb`: contains predicting on `testb` dataset and
    post-processing the output to fit the submission format
    
- log: contains experiment logs
    - trained: contains trained models for ner_crf_reg
    - progress_rnn1_10_09_12_12: contains the most up-to-date LSTM+FC model logs
    - other folders can be mostly for archive purpose
- data: contains original CONLL dataset 
    [todo: remove this for copyright]
    
- report: contains the report writeup and images required for the writeup
    - `report.md`: this is the report for my experiments
    - images: a folder that contains figures for the report 
    - predictions: contains predictions on `testb` dataset
        - `testb_crf_pred.txt`: predictions made by CRF model (Part 1)
        - `testb_rnn1_runn2_pred.txt`: predictions made by LSTM+FC model (Part 2)
        
    
    