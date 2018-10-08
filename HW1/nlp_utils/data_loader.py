# hjsong
# reference: github.com/cs230/nlp
import random,os,joblib
import numpy as np
import torch
from nlp_utils.model_utils import Params

import pdb # for debug
class SentenceDatasetsLoader(object):
    """
    Loads a mini-batch of data at each iterations. 
    Stores the following properties:
    - dataset_params
    - word2idx: word to index mapping
    - tag2idx: tag to index mapping
    
    Args:
    - data_dir (str): path to the directory containing the dataset
    - word2idx_file (str): filename to `word2idx` file in `data_dir`
    - tag2idx_file (str): filename to `tag2idx` file in `data_dir`
    - data_types (optional, list of str): a list of string indicating
        data types. Default is `['train', 'dev', 'testa']`
    """
    def __init__(self, data_dir, word2idx_file, tag2idx_file, data_types=None):
        self.data_dir = data_dir
        
        word2idx_path = os.path.join(data_dir, word2idx_file)
        tag2idx_path = os.path.join(data_dir, tag2idx_file)
        self.word2idx = joblib.load(word2idx_path)
        self.tag2idx = joblib.load(tag2idx_path)
        self.vocab_size = len(self.word2idx)
        self.n_tags = len(self.tag2idx)
        
        if data_types is None:
            data_types = ['train', 'dev', 'testa']
        self.data_types = data_types
        
    def load_sentences_labels(self, sentences_path, labels_path, d):
        """
        Load sentences and labels for this dataset to the input dictionary d
        """
        d['data'] = joblib.load(sentences_path)
        d['labels'] = joblib.load(labels_path)
        d['size'] = len(d['data'])
        
        
    def load_datasets(self,types):
        """
        Load dataset(s) from data_dir.
        Args:
        - data_dir (str): path to the directory that contains dataset files
            We assume the dataset file is after mapping each word to its index in the vocab.
            
        - types (list): a list of string(s) which is one of 'train', 'dev', 'testa'
        
        Returns:
        - dataset (dict): contains the sentences and labels for each type in types
        """ 
        for t in types:
            if t not in self.data_types:
                raise ValueError(f'Check your types. They must come from {self.data_types}')
                
        datasets = {}
        for split in self.data_types:
            if split in types:
                sentences_path = os.path.join(self.data_dir, split+"_sentences.sav")
                labels_path = os.path.join(self.data_dir, split+"_labels.sav")
                print(sentences_path, "\n", labels_path)
                                     
                datasets[split] = {}
                self.load_sentences_labels(sentences_path, labels_path, datasets[split])
                print("Loaded ", split)
        return datasets
    
    def data_iterator(self, dataset, params, shuffle=False):
        """
        Returns a generator that yields a mini-batch of data (sentences and labels).
        It iteratates once over the data.
        
        Args:
        - dataset (dict): a dictionary with keys of 'data', 'labels', 'size'
        - params (Params): hyperparams of the training. Must have the following fields:
            - batch_size (int)
            - cuda (bool)
            - pad_word (strng)
        - shuffle (bool): to shuffle the mini-batch or not
        
        Yields:
        - batch_data (torch.LongTensor): word indices of size batch_size * seq_len 
        - batch_labels (torch.LongTensor): tag indices of size batch_size * seq_len
        """
        order = list(range(dataset['size']))
        if shuffle:
            random.seed(0)
            random.shuffle(order)
        
        # Single iteration over data
#         pdb.set_trace()
        for i in range( (dataset['size']+1)//params.batch_size ):
            # Get a batch of sentences and tags 
            batch_sentences = [dataset['data'][i] for i in order[i*params.batch_size: (i+1)*params.batch_size]]
            batch_tags = [dataset['labels'][i] for i in order[i*params.batch_size: (i+1)*params.batch_size]]
            
#             ## todo: debug
#             pdb.set_trace()
#             idx2word = {i:w for w,i in self.word2idx.items()}
#             for _sent, _tag in zip(batch_sentences, batch_tags):
#                 print([idx2word[_idx] for _idx in _sent])
#                 print(_tag)
#             ## end of debug
            
            
            # Perform the two modification mentioned above
            # 1. Append PAD words so that all sentences in this batch are of the same length
            # 2. Mark unseen word's tag as -1 (for evaluation)
            batch_max_len = max([len(s) for s in batch_sentences])

            # Intialize new batch matrix
            ## Use -1 for initial tags to differentiate it with tags from PAD_WORDs
            batch_data = self.word2idx[params.pad_word]*np.ones((len(batch_sentences), batch_max_len))
            batch_labels = -1*np.ones((len(batch_sentences), batch_max_len))
#             print(type(batch_data), type(batch_labels))


            # Fill in the matrix with current batch sentences and labels
            for sidx in range(len(batch_sentences)):
                curr_len = len(batch_sentences[sidx])
                batch_data[sidx][:curr_len] = batch_sentences[sidx]
                batch_labels[sidx][:curr_len] = batch_tags[sidx]

            # Convert to torch.LongTensors (since each entry is an index)
#             batch_data = torch.LongTensor(batch_data), torch.LongTensor(batch_labels)
            batch_data, batch_labels = torch.from_numpy(batch_data), torch.from_numpy(batch_labels)
            batch_data = batch_data.type(torch.LongTensor)
            batch_labels = batch_labels.type(torch.LongTensor)
#             print(type(batch_data), type(batch_labels))

            # If gpu available
            if params.cuda:
                batch_data, batch_labels = batch_data.cuda(), batch_labels.cuda()
            yield batch_data, batch_labels

