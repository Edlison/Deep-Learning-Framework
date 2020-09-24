# @Author  : Edlison
# @Date    : 7/30/20 11:56
import torch
import numpy as np


class Model(torch.nn.Module):
    """
    need net, optimizer, criterion.
    """

    def __init__(self):
        super().__init__()

    def forward(self, X):
        ...
        raise NotImplementedError

    def _load_dict(self, path):
        # path = '../data/cache/dict/dict.tempo'
        with open(path, 'r', encoding='utf-8') as f:
            dict_ = eval(f.read())
        return dict_

    def load_pretrained(self, dict_path, path, word_dim):
        words_dict = self._load_dict(dict_path)
        vecs = np.random.normal(0.0, 0.9, [len(words_dict), word_dim])
        with open(path, 'r') as f:
            for line in f:
                line = line.split()
                if line[0] in words_dict:
                    vecs[words_dict[line[0]]] = np.asarray(line[1:], dtype='float32')
        return vecs
