# @Author  : Edlison
# @Date    : 7/29/20 23:27

import torch
from core.v1.pre import TrainLoader
from core.v1.pre import EvalLoader
from core.v1.pre import TestLoader
from core.v1.model import Model
from core.v1.train import Trainer
from core.v1.test import Tester
import string
import time

train_path = '../data/imdb/train_data.txt'
eval_path = '../data/imdb/imdb_test_new_withlabel.txt'
test_path = '../data/imdb/imdb_test_new_withlabel.txt'
stopwords_path = '../data/stopwords/stop_en.txt'
pretrained_path = '../data/pretrained/glove.6B/glove.6B.50d.txt'


class train_ImdbLoader(TrainLoader):
    def __init__(self, dict_size: int, features_size: int, gen_dict_by: str, batch_size: int, rm_stop: str):
        super().__init__(dict_size, features_size, gen_dict_by, batch_size, rm_stop)

    def read(self):
        samples = []
        labels = []
        with open(train_path, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                samples.append(line.split('<SEP>')[0].strip().lower())  # 去除前后空格 文本转为小写
                labels.append(line.split('<SEP>')[1].strip().lower())
        self.samples = samples
        self.labels = labels

    def clean(self):
        samples = []
        for sample in self.samples:
            rm_char = string.punctuation  # 拿到全部英文标点符号和(数字?)
            for each in rm_char:  # 移除英文标点符号
                sample = sample.replace(each, '')
            sample = sample.split(' ')
            samples.append(sample)
        self.samples = samples


class eval_ImdbLoader(EvalLoader):
    def __init__(self, words_dict_path, features_size, batch_size: int, rm_stop: str):
        super().__init__(words_dict_path, features_size, batch_size, rm_stop)

    def read(self):
        samples = []
        labels = []
        with open(eval_path, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                samples.append(line.split('<SEP>')[0].strip().lower())  # 去除前后空格 文本转为小写
                labels.append(line.split('<SEP>')[1].strip().lower())
        self.samples = samples
        self.labels = labels

    def clean(self):
        samples = []
        for sample in self.samples:
            rm_char = string.punctuation  # 拿到全部英文标点符号和(数字?)
            for each in rm_char:  # 移除英文标点符号
                sample = sample.replace(each, '')
            sample = sample.split(' ')
            samples.append(sample)
        self.samples = samples


class test_ImdbLoader(TestLoader):
    def __init__(self, words_dict_path, features_size, batch_size, rm_stop: str):
        super().__init__(words_dict_path, features_size, batch_size, rm_stop)

    def read(self):
        samples = []
        with open(test_path, 'r') as f:
            for line in f:
                line = line.split('\n')[0]
                samples.append(line.split('<SEP>')[0].strip().lower())  # 去除前后空格 文本转为小写
        self.samples = samples

    def clean(self):
        samples = []
        for sample in self.samples:
            rm_char = string.punctuation  # 拿到全部英文标点符号和(数字?)
            for each in rm_char:  # 移除英文标点符号
                sample = sample.replace(each, '')
            sample = sample.split(' ')
            samples.append(sample)
        self.samples = samples


class RNNModel(Model):
    def __init__(self, dict_size, word_dim, layer_num, hidden_size, output_num):  # 需要手动传dict_size 可优化
        super().__init__()
        vecs = self.load_pretrained(dict_path, pretrained_path, 50)
        self.embedding_layer = torch.nn.Embedding(dict_size, word_dim)
        self.embedding_layer.from_pretrained(torch.from_numpy(vecs))
        self.embedding_layer.weight.requires_grad = False

        self.rnn = torch.nn.LSTM(word_dim, hidden_size, layer_num, batch_first=True)
        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.4),
            torch.nn.Linear(hidden_size, output_num)
        )
        # self.fc = torch.nn.Linear(hidden_size, output_num)

    def forward(self, X):
        # print('forward', X, X.shape)  # [500, 30]
        # X = X.permute(1, 0)
        # print('forward', X, X.shape)  # [30, 500]
        X = self.embedding_layer(X)
        # outs, _ = self.rnn(X)  # 通过embedding层变为3d [batch, seq, wordvec]
        # outs = self.fc(outs[-1])
        outs, (hs, hc) = self.rnn(X)
        outs = self.fc(outs[:, -1, :])
        return outs


def accuracy(pred_y, true_y):
    """
    计算二类和多类的准确率
    :param pred_y: 预测结果
    :param true_y: 真实结果
    :return:
    """
    if isinstance(pred_y[0], list):
        pred_y = [item[0] for item in pred_y]
    corr = 0
    for i in range(len(pred_y)):
        if pred_y[i] == true_y[i]:
            corr += 1
    acc = corr / len(pred_y) if len(pred_y) > 0 else 0
    return acc


def load_predict(path):
    predict = []
    with open(path, 'r') as f:
        for line in f:
            line = line.split('\n')[0]
            predict.append(line.split('<SEP>')[0].strip().lower())  # 去除前后空格 文本转为小写
    predict = [int(i) for i in predict]
    return predict


def load_true(path):
    true = []
    with open(path, 'r') as f:
        for line in f:
            line = line.split('\n')[0]
            true.append(line.split('<SEP>')[1].strip().lower())  # 去除前后空格 文本转为小写
    true = [int(i) for i in true]
    return true


if __name__ == '__main__':
    dict_size = 1000
    features_size = 100
    batch_size = 500
    word_dim = 50
    dict_path = '../data/cache/dict/dict.tempo'
    model_path = '../data/cache/model/model.pt'
    test_labels_path = '../data/output/imdb_v1_out_2.txt'

    time1 = time.time()

    print('- train loader exe start!')
    trainLoader = train_ImdbLoader(dict_size=dict_size,
                                   features_size=features_size,
                                   gen_dict_by='frequency',
                                   batch_size=batch_size,
                                   rm_stop=stopwords_path)
    print('- dict save start!')
    trainLoader.save_dict(dict_path)
    print('- eval loader exe start!')
    evalLoader = eval_ImdbLoader(words_dict_path=dict_path,
                                 features_size=features_size,
                                 batch_size=batch_size,
                                 rm_stop=stopwords_path)
    print('- rnnModel start!')
    rnnModel = RNNModel(dict_size=dict_size,
                        word_dim=word_dim,
                        layer_num=1,
                        hidden_size=128,
                        output_num=2)
    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(rnnModel.parameters())
    print('- trainer exe start!')
    trainer = Trainer(train_iter=trainLoader.get_data(),
                      eval_iter=evalLoader.get_data(),
                      model=rnnModel,
                      criterion=criterion,
                      optimizer=optimizer,
                      epoch=2)
    print('- trainer.start start!')
    trainer.start()
    print('- trainer save model start!')
    trainer.save_model(model_path)
    print('- testLoader exe start!')
    testLoader = test_ImdbLoader(words_dict_path=dict_path,
                                 features_size=features_size,
                                 batch_size=batch_size,
                                 rm_stop=stopwords_path)
    print('- tester exe start!')
    tester = Tester(test_iter=testLoader.get_data(),
                    model_path=model_path)
    print('- test labels test start!')
    tester.test()
    print('- labels export start!')
    tester.export(test_labels_path)
    time2 = time.time()
    print('total', time2-time1)

    pred_path = '../data/output/imdb_v1_out_2.txt'
    true_path = '../data/imdb/imdb_test_new_withlabel.txt'

    pred = load_predict(pred_path)
    true = load_true(true_path)

    a = accuracy(pred, true)
    print('acc:', a)
    print(pred[:10])
    print(true[:10])