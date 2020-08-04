from torch.utils.data import Dataset
import jieba
import torch
from tqdm import tqdm

def read_cnews_data(path):
    labels = []
    datas = []
    label_num = {}
    with open(path, 'r', encoding='utf-8') as file:
        for i, line in enumerate(file):
            if i == 100: break
            if line[:2] not in label_num:
                label_num[line[:2]] = len(label_num)
            labels.append(label_num[line[:2]]) # 0, 1, 2, 3
            datas.append(line[3:])
    return datas, labels, len(label_num)


class MyDataset(Dataset):
    def __init__(self, path:str):
        self.datas, self.labels, self.label_num = read_cnews_data(path)
        self.data2token()


    def token2seq(self, vocab, padding_len):
        for i, data in enumerate(self.datas):
            if len(self.datas[i]) < padding_len:
                self.datas[i] += [vocab.word2seq(vocab.padding_word)] * (padding_len - len(self.datas[i]))
            elif len(self.datas[i]) > padding_len:
                self.datas[i] = self.datas[i][:padding_len]
            for j in range(padding_len):
                self.datas[i][j] = vocab.word2seq(self.datas[i][j])
            self.datas[i] = torch.tensor(self.datas[i], dtype=torch.long)
        self.labels = torch.tensor(self.labels, dtype=torch.long)

    def data2token(self):
        self.avg_len = 0
        for i, data in enumerate(tqdm(self.datas)):
            self.datas[i] = jieba.lcut(data)
            self.avg_len += len(self.datas[i])
        self.avg_len /= len(self)
        print(f'the average len is {self.avg_len}')


    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item:int):
        return self.datas[item], self.labels[item]

if __name__ == '__main__':
    from test._vocab1 import Vocab
    train_path = r'cnews/cnews.train.txt'
    train_dataset = MyDataset(train_path)
    vocab = Vocab(train_dataset.datas, 5000)
    print(vocab.word2seq('天气'))


