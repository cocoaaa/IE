import numpy as np


# pytorch imports 
import torch
import torch.autograd as autograd

from torch import Tensor
import torch.nn as nn
import torch.optim as optim

# set a random seed
# torch.manual_seed(10);

# model saving and inspection
# import joblib
# import eli5
# from datetime import datetime

import pdb # debugging


class BILSTM_CRF(nn.Module):
    def __init__(self, dim_embedding, dim_hidden, vocab_size, tag2idx,
                 n_lstm_layers=1):
        super(BILSTM_CRF, self).__init__()
        self.dim_embedding = dim_embedding
        self.dim_hidden = dim_hidden
        self.n_lstm_layers = n_lstm_layers
        self.vocab_size = vocab_size
        self.tag2idx = tag2idx
        self.output_size = len(tag2idx) #n_tags
        
        self.embedding = nn.Embedding(vocab_size, dim_embedding)
        self.lstm = nn.LSTM(dim_embedding, dim_hidden//2,
                            num_layers=n_lstm_layers, bidirectional=True)
        
        # output of biLSTM to tag
        self.hidden2tag = nn.Linear(self.dim_hidden, self.output_size)
        
        # Transition matrix for CRF
        ## T(i,j) = log_prob(tag_j -> tag_i). Note **from** j **to** i
        self.transitions = nn.Parameter(torch.randn(self.output_size, self.output_size))
        ## Never transit tag_i -> START_TAG and END_TAG -> tag_i
        self.transitions.data[tag2idx[START_TAG], :] = -1e6
        self.transitions.data[:, tag2idx[STOP_TAG],] = -1e6
        
        # Initial hidden layers
        self.hidden = self.init_hidden()
    
    def init_hidden(self):
        return [torch.randn(2,1,self.dim_hidden//2), 
                torch.randn(2,1,self.dim_hidden//2)]
    def _viterbi_forward(self, feats):
        """
        Args:
        - feats (tensor): output feature vector from LSTM layer
        """
        # Forward pass to compute the partition function
        init_alphas = torch.full((1, self.output_size), -1e-6)
        
        # Fill in the entries for START_TAG
        init_alphas[0][self.tag2idx[START_TAG]] = 0.0
        
        # For automatic backprop
        forward_var = init_alphas
        
        # Iterate through the sequence
        for feat in feats:
            alphas = []
            for tag in range(self.output_size):
                emit_score = torch.full((1,self.output_size), feat[tag].item())
                
                # jth entry of trans_score is the score of transitioning from j
                # to tag
                trans_score = self.transitions[tag].view(1,-1)
                
                tag_var = forward_var + emit_score + trans_score
                alphas.append(log_sum_exp(tag_var).view(1))
            forward_var = torch.cat(alphas).view(1,-1)
        terminal_var = forward_var + self.transitions[self.tag2idx[STOP_TAG]]
        alpha = log_sum_exp(terminal_var)
        return alpha
    
    def _get_lstm_features(self, sentence):
        """
        Args:
        - sentence (torch.LongTensor): a 1D LongTensor of word indices
        """
        self.hidden = self.init_hidden()
        embedding = self.embedding(sentence).view(len(sentence), 1, -1)
        
        # Forward through LSTM
        lstm_out, self.hidden = self.lstm(embedding, self.hidden)
        lstm_out = lstm_out.view(len(sentence), self.dim_hidden)
        
        # Forward the feature vector from LSTM to output activation
        # through another linear layer
        lstm_feats = self.hidden2tag(lstm_out)
        return lstm_feats
    
    def _score_sentence(self, feats, tags):
        # Returns the score of the input tag sequence
        score = torch.zeros(1)
        
        # Prepend the START_TAG
        tags = torch.cat([torch.tensor([self.tag2idx[START_TAG]], dtype=torch.long),
                          tags])
        
        for i, feat in enumerate(feats):
            score += self.transitions[tags[i+1], tags[i]] + feat[tags[i+1]]
      
        # Lastly, add the transition score to the STOP_TAG
        score += self.transitions[self.tag2idx[STOP_TAG], tags[-1]]
        return score
    
    def _viterbi_decode(self, feats):
        backpointers = []
        
        # Initialize the viterbi vars in log domain
        init_vvars = torch.full( (1, self.output_size), -1e6 )
        init_vvars[0][self.tag2idx[START_TAG]] = 0 # initial starting point
        
        # Forward viterbi algorithm
        forward_var = init_vvars
        for feat in feats:
            
            bptrs = [] # backpointers for this time step
            vvars = [] # viberbi vars for this time step
            for tag in range(self.output_size):
                tag_var = forward_var + self.transitions[tag]
                _, best_tid = torch.max(tag_var,1)
                best_tid = best_tid.item()
                bptrs.append(best_tid)
                vvars.append(tag_var[0][best_tid].view(1))
            # Add in the emission scores 
            forward_var = (torch.cat(vvars) + feat).view(1,-1)
            backpointers.append(bptrs)
            
        # Add transition to STOP_TAG
        terminal_var = forward_var + self.transitions[self.tag2idx[STOP_TAG]]
        _, best_tid =  torch.max(terminal_var,1)
        best_tid = best_tid.item()
        path_score = terminal_var[0][best_tid]
        
        # Backtrace the backpointers to find the best path
        best_path = [best_tid]
        for bptrs in reversed(backpointers):
            best_tid = bptrs[best_tid]
            best_path.append(best_tid)
        
        # Remove the START_TAG 
        start = best_path.pop()
        assert (start == self.tag2idx[START_TAG])
        
        # Reverse the path order
        best_path.reverse()
        
        return path_score, best_path

    def neg_log_likelihood(self, sentence, tags):
        # Computes negative log likelihood of having tags given the sentence
        feats = self._get_lstm_features(sentence)
        forward_score = self._viterbi_forward(feats)
        score = self._score_sentence(feats, tags)
        return forward_score - score
    
    def forward(self, sentence):
        # Returns the best path score and the best path, given the setence
        
        # features from the BILSTM
        lstm_feats = self._get_lstm_features(sentence)
        
        # Find the best path and its score, given the input sentence
        best_score, tag_seq = self._viterbi_decode(lstm_feats)
        return best_score, tag_seq