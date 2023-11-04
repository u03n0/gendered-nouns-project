from collections import defaultdict

import torch
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim
from torch.utils.data import Dataset


class LanguageDataset(Dataset):
    """ Custom Language dataset 
    
    """

    def __init__(self, data):
        self.data = data
        self.char2idx = {}
        self.labels = []
        self.longest_noun = 0

        idx = 0
        for tup in data:
            noun, label = tup.strip().split(",")
            if len(noun) > self.longest_noun:
                self.longest_noun = len(noun)
            for char in noun:
                if char not in self.char2idx:
                    self.char2idx[char] = idx
                    idx +=1
            if label not in self.labels:
                self.labels.append(label)
        self.idx2char = {v: k for k, v in self.char2idx.items()}
        

    def __len__(self):
        return len(self.data)
    
    def __getitem__(self, idx):
        pass



class GenderedLSTM(nn.Module):

    def __init__(self, traingenerator, embedding_dim, hidden_dim, vocab_size, tagset_size):
        super(GenderedLSTM, self).__init__()
        self.embedding_dim = embedding_dim
        self.hidden_dim = hidden_dim
        self.vocab_size = vocab_size
        self.tagset_size = tagset_size
        self.allocate_params(traingenerator)


    def allocate_params(self, datagenerator):

        self.word_embeddings = nn.Embedding(self.vocab_size, self.embedding_dim)

        self.lstm = nn.LSTM(self.embedding_dim, self.hidden_dim)

        self.hidden2tag = nn.Linear(self.hidden_dim, self.tagset_size)


    def foward(self, sequence):
        embeds = self.word_embeddings(sequence)
        lstm_out, _ = self.lstm(embeds.view(len(sequence), 1, -1))
        tag_space = self.hidden2tag(lstm_out.view(len(sequence), -1))
        tag_score = F.log_softmax(tag_space, dim=1)
        return tag_score
    
    def train(self, traingenerator, validgenerator, epochs, batch_size, device='cuda', learning_rate=0.1):

        loss_function = nn.NLLLoss()
        optimizer = optim.Adam(self.parameters(), lr=learning_rate)
        # with torch.no_grad():
        #     inputs = 