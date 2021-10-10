"""
author-gh: @adithya8
editor-gh: ykl7
"""

import math 

import numpy as np
from numpy.core.defchararray import add
import torch
from torch._C import device
import torch.nn as nn
import torch.nn.functional as F


sigmoid = lambda x: 1/(1 + torch.exp(-x))

class WordVec(nn.Module):
    def __init__(self, V, embedding_dim, loss_func, counts):
        super(WordVec, self).__init__()
        self.center_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.center_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.center_embeddings.weight.data[self.center_embeddings.weight.data<-1] = -1
        self.center_embeddings.weight.data[self.center_embeddings.weight.data>1] = 1

        self.context_embeddings = nn.Embedding(num_embeddings=V, embedding_dim=embedding_dim)
        self.context_embeddings.weight.data.normal_(mean=0, std=1/math.sqrt(embedding_dim))
        self.context_embeddings.weight.data[self.context_embeddings.weight.data<-1] = -1 + 1e-10
        self.context_embeddings.weight.data[self.context_embeddings.weight.data>1] = 1 - 1e-10
        
        self.loss_func = loss_func
        self.counts = counts

    def forward(self, center_word, context_word):

        if self.loss_func == "nll":
            return self.negative_log_likelihood_loss(center_word, context_word)
        elif self.loss_func == "neg":
            return self.negative_sampling(center_word, context_word)
        else:
            raise Exception("No implementation found for %s"%(self.loss_func))
    
    def negative_log_likelihood_loss(self, center_word, context_word):
        # get embedding of center and context words
        emb_input = self.center_embeddings(center_word)    
        emb_context = self.context_embeddings(context_word)  
        # calculate NLL objective function
        sum = torch.sum(torch.multiply(emb_input,emb_context),axis=1)
        log = torch.log(torch.sum(torch.exp(torch.matmul(emb_input,emb_context.t())),axis=1))
        # take mean of all the loss
        loss = torch.mean(torch.sub(log,sum))
       

        return loss
    
    def negative_sampling(self, center_word, context_word):

        # convert counts numpy array to tensor 
        counts_matrix = torch.from_numpy(self.counts).to(torch.device('cuda:0'))

        # to get the probability distribution matrix, raise every elemnt to (3/4) power 
        # according to the paper mentioned in readme file
        counts_matrix = torch.pow(counts_matrix,0.75).to(torch.device('cuda:0'))

        # normalize with sum
        counts_matrix = torch.div(counts_matrix,(torch.sum(counts_matrix))).to(torch.device('cuda:0'))

        #draw samples using multinomial distribution
        nwords = torch.multinomial(counts_matrix,5, replacement=False).to(torch.device('cuda:0'))

        # embedding of center ,context, negative words
        emb_nwords = (self.context_embeddings(nwords)).to(torch.device('cuda:0'))
        emb_input = self.center_embeddings(center_word)
        emb_context = self.context_embeddings(context_word)

        # calculate loss objective function
        logval = torch.log(sigmoid(torch.matmul(emb_input,emb_context.t()))).to(torch.device('cuda:0'))
        sumlogval = torch.sum(torch.log(sigmoid(torch.mul(-1,torch.matmul(emb_input,emb_nwords.t())))),axis=1).to(torch.device('cuda:0'))
        
        # take mean of loss
        loss = torch.mean(torch.add(logval,sumlogval)) * (-1)


        return loss

    def print_closest(self, validation_words, reverse_dictionary, top_k=8):
        print('Printing closest words')
        embeddings = torch.zeros(self.center_embeddings.weight.shape).copy_(self.center_embeddings.weight)
        embeddings = embeddings.data.cpu().numpy()

        validation_ids = validation_words
        norm = np.sqrt(np.sum(np.square(embeddings),axis=1,keepdims=True))
        normalized_embeddings = embeddings/norm
        validation_embeddings = normalized_embeddings[validation_ids]
        similarity = np.matmul(validation_embeddings, normalized_embeddings.T)
        for i in range(len(validation_ids)):
            word = reverse_dictionary[validation_words[i]]
            nearest = (-similarity[i, :]).argsort()[1:top_k+1]
            print(word, [reverse_dictionary[nearest[k]] for k in range(top_k)])            