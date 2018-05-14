import torch
import torch.nn as nn

import data_loader

class LanguageModel(nn.Module):

    def __init__(self, vocab_size, word_vec_dim = 512, hidden_size = 512, n_layers = 4, dropout_p = .2, max_length = 255):
        self.vocab_size = vocab_size
        self.word_vec_dim = word_vec_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        super(LanguageModel, self).__init__()

        self.emb = nn.Embedding(vocab_size, word_vec_dim, padding_idx = data_loader.PAD)
        self.rnn = nn.LSTM(word_vec_dim, hidden_size, n_layers, batch_first = True, dropout = dropout_p)
        self.out = nn.Linear(hidden_size, vocab_size, bias = True)
        self.log_softmax = nn.LogSoftmax(dim = 2)

    def forward(self, x):
        # |x| = (batch_size, length)
        x = self.emb(x) 
        # |x| = (batch_size, length, word_vec_dim)
        x, (h, c) = self.rnn(x) 
        # |x| = (batch_size, length, hidden_size)
        x = self.out(x) 
        # |x| = (batch_size, length, vocab_size)
        y_hat = self.log_softmax(x)

        return y_hat