# @Author  : Edlison
# @Date    : 8/1/20 00:47

from core.v1 import model
from core.v1 import pre
import re
import jieba
import torch
import time
from tqdm import tqdm
from core.v1 import train
from core.v1 import test

cnews_train = '../data/cnews/cnews.train.txt'
cnews_eval = '../data/cnews/cnews.val.txt'


# Implement
class CnewsTrainLoader(pre.TrainLoader):
    def __init__(self, dict_size: int, features_size: int, gen_dict_by: str, batch_size: int, rm_stop: str):
        super().__init__(dict_size, features_size, gen_dict_by, batch_size, rm_stop)

    def read(self):
        samples = []
        labels = []
        tag = {}
        with open(cnews_train, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split('\n')[0]
                samples.append(line.split('\t', 1)[1].strip())
                label = line.split('\t', 1)[0].strip()
                if label not in tag:
                    tag[label] = len(tag)
                labels.append(tag[label])
        self.samples = samples
        self.labels = labels
        self.tag = tag

    def clean(self):
        pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
        samples = []
        for sample in tqdm(self.samples):
            sample = re.sub(pattern, '', sample)
            sample = ''.join(sample.split())
            sample = jieba.lcut(sample)
            samples.append(sample)
        self.samples = samples

    def getTag(self):
        return self.tag


class CnewsEvalLoader(pre.EvalLoader):
    def __init__(self, words_dict_path, features_size, batch_size: int, rm_stop: str, tag):
        self.tag = tag
        super().__init__(words_dict_path, features_size, batch_size, rm_stop)

    def read(self):
        samples = []
        labels = []
        tag = self.tag
        with open(cnews_train, 'r', encoding='utf-8') as f:
            for line in f:
                line = line.split('\n')[0]
                samples.append(line.split('\t', 1)[1].strip())
                label = line.split('\t', 1)[0].strip()
                labels.append(tag[label])
        self.samples = samples
        self.labels = labels

    def clean(self):
        pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")
        samples = []
        for sample in tqdm(self.samples):
            sample = re.sub(pattern, '', sample)
            sample = ''.join(sample.split())
            sample = jieba.lcut(sample)
            samples.append(sample)
        self.samples = samples


class CnewsTestLoader(pre.TestLoader):
    def __init__(self, words_dict_path, features_size, batch_size: int, rm_stop: str):
        super().__init__(words_dict_path, features_size, batch_size, rm_stop)

    def read(self):
        ...

    def clean(self):
        ...


class Model(model.Model):
    def __init__(self, dict_path, pretrained_path, dict_size, embedding_dim, hidden_size, num_layer, num_output):
        super().__init__()

        vector = self.load_pretrained(dict_path, pretrained_path, embedding_dim)

        self.embedding = torch.nn.Embedding(dict_size, embedding_dim)
        self.embedding.from_pretrained(torch.from_numpy(vector))
        self.embedding.weight.requires_grad = False

        self.lstm = torch.nn.LSTM(embedding_dim, hidden_size, num_layer, batch_first=True, bidirectional=True)

        self.fc = torch.nn.Sequential(
            torch.nn.Dropout(0.5),
            torch.nn.Linear(hidden_size * 2, hidden_size),
            torch.nn.Linear(hidden_size, num_output)
        )

    def forward(self, X):
        X = X.permute(1, 0)
        embed = self.embedding(X)
        outs, _ = self.lstm(embed)
        output = self.fc(outs[-1])
        return output

# Run
if __name__ == '__main__':
    dict_size = 1000
    feature_size = 100
    batch_size = 500
    embedding_dim = 300
    hidden_size = 128
    num_layer = 1
    num_output = 10
    pretrained_path = '../data/pretrained/sgns.wiki.word'
    stop_path = '../data/stopwords/stop_zh.txt'
    dict_path = '../data/cache/dict/cnews.tempo'
    model_path = '../data/cache/dict/cnews.pt'

    time0 = time.time()
    trainLoader = CnewsTrainLoader(
        dict_size=dict_size,
        features_size=feature_size,
        gen_dict_by='frequency',
        batch_size=batch_size,
        rm_stop=stop_path
    )
    time1 = time.time()
    print(f'trainLoader completed cost {time1 - time0} s')

    trainLoader.save_dict(dict_path)

    evalLoader = CnewsEvalLoader(
        words_dict_path=dict_path,
        features_size=feature_size,
        batch_size=batch_size,
        rm_stop=stop_path,
        tag=trainLoader.getTag()
    )
    print('evalLoader completed!')

    model = Model(
        dict_path=dict_path,
        pretrained_path=pretrained_path,
        dict_size=dict_size,
        embedding_dim=embedding_dim,
        hidden_size=hidden_size,
        num_layer=num_layer,
        num_output=num_output
    )
    print('model completed!')

    criterion = torch.nn.CrossEntropyLoss()
    optimizer = torch.optim.Adam(model.parameters())

    train.Trainer(train_iter=trainLoader.get_data(),
                  eval_iter=evalLoader.get_data(),
                  model=model,
                  criterion=criterion,
                  optimizer=optimizer,
                  epoch=10).start()


