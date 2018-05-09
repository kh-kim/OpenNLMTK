import torch
import torch.nn as nn

class LanguageModel(nn.Module):

    def __init__(self, vocab_size, word_vec_dim = 512, hidden_size = 512, n_layers = 4, dropout_p = .2, max_length = 255):
        self.vocab_size = vocab_size
        self.word_vec_dim = word_vec_dim
        self.hidden_size = hidden_size
        self.n_layers = n_layers
        self.dropout_p = dropout_p
        self.max_length = max_length

        super(LanguageModel, self).__init__()

    def forward(self, x):

        return y_hat
