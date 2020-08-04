import torch
from torch.utils.data import DataLoader
from torch import nn, optim
from test._data1 import MyDataset
from test._vocab1 import Vocab
from test._model1 import RNN, read_pretrained_wordvec
from tqdm import tqdm




train_dataset = MyDataset(r'../data/cnews/cnews.train.txt')
test_dataset = MyDataset(r'../data/cnews/cnews.test.txt')
vocab = Vocab(train_dataset.datas, 100)
train_dataset.token2seq(vocab, 100)
test_dataset.token2seq(vocab, 100)
train_dataset = DataLoader(train_dataset, batch_size=8, shuffle=True)
test_dataset = DataLoader(test_dataset, batch_size=8)

net = RNN(read_pretrained_wordvec(r'../data/pretrained/glove.6B/glove.6B.50d.txt', vocab, 50), len(vocab), 50, 1, 16, 10)
optimizer = optim.Adam(net.parameters(), lr=1e-3)
criterion = nn.CrossEntropyLoss()


def train(epoch):
    def evaluate():
        net.eval()
        correct = 0
        all = 0
        with torch.no_grad():
            for (x, y) in tqdm(test_dataset):

                logits = net(x)
                logits = torch.argmax(logits, dim=-1)
                correct += torch.sum(logits.eq(y)).float().item()
                all += y.size()[0]
        print(f'evaluate done! acc {correct / all:.5f}')

    for ep in range(epoch):
        print(f'epoch {ep} start')
        net.train()
        for (x, y) in tqdm(train_dataset):

            # print('before net', x, x.shape)  # [8, 100]
            logits = net(x)
            # print('after net logits', logits, logits.shape)  # [8, 10]
            # print('after net y', y, y.shape)  # [8]
            loss = criterion(logits, y)
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        evaluate()

if __name__ == '__main__':
    train(10)