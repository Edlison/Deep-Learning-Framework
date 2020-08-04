import torch
from torch import nn
import numpy as np
from test._vocab1 import Vocab


def read_pretrained_wordvec(path, vocab:Vocab, word_dim):
    vecs = np.random.normal(0.0, 0.9, [len(vocab), word_dim])
    with open(path, 'r') as file:
        for line in file:
            line = line.split()
            if line[0] in vocab.vocab:
                vecs[vocab.word2seq(line[0])] = np.asarray(line[1:], dtype='float32')
    return vecs


class RNN(nn.Module):
    def __init__(self, vecs, vocab_size, word_dim, num_layer, hidden_size, label_num):
        super(RNN, self).__init__()
        self.embedding_layer = nn.Embedding(vocab_size, word_dim)
        self.embedding_layer.from_pretrained(torch.from_numpy(vecs))
        self.embedding_layer.weight.requires_grad = False

        self.rnn = nn.LSTM(word_dim, hidden_size, num_layers=num_layer)
        self.fc = nn.Sequential(
            nn.Dropout(0.4),
            nn.Linear(hidden_size, label_num)
        )

    def forward(self, X):
        # print('forward', X, X.shape)  # [8, 100]
        X = X.permute(1, 0) # [batch, seq, word_vec] -> [seq, batch, word_vec]
        # print('forward', X, X.shape)  # [100, 8]
        X = self.embedding_layer(X)
        outs, _ = self.rnn(X)
        logits = self.fc(outs[-1])
        return logits





if __name__ == '__main__':
    x = np.random.normal(0.0, 0.9, [2, 10])
    print(x, type(x))