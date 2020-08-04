# @Author  : Edlison
# @Date    : 7/27/20 02:57

import torch
from torch.utils.data import Dataset, DataLoader
import numpy as np
import json
from core.processor.wordsbag import DataProcessor

def load():
    with open('../config/imdb_config.json', 'r') as f:
        config = json.load(f)

    dp = DataProcessor(config)
    dp.train_data_path = '../' + dp.train_data_path
    dp.stop_words_path = '../' + dp.stop_words_path
    dp.eval_data_path = '../' + dp.eval_data_path
    print('start imdb')
    dp.imdb()
    dp.load_en_eval()
    dp.split_eval_words()
    print('gen features')
    train_samples = torch.tensor(dp.pow_boolean(dp.words_dict, dp.train_X))
    label_0 = [int(_) for _ in dp.train_y]
    train_labels = torch.tensor(label_0)

    eval_samples = torch.tensor(dp.pow_boolean(dp.words_dict, dp.eval_X))
    label_1 = [int(_) for _ in dp.eval_y]
    eval_labels = torch.tensor(label_1)
    print('start iter')

    return train_samples.float(), train_labels, eval_samples.float(), eval_labels

class imdbDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.samples[item], self.labels[item]


class MLP(torch.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = torch.nn.Linear(input_num, hidden_num_1)
        self.fc2 = torch.nn.Linear(hidden_num_1, hidden_num_2)
        self.fc3 = torch.nn.Linear(hidden_num_2, output_num)

    def forward(self, din):
        din = din.view(-1, input_num)
        dout = torch.nn.functional.relu(self.fc1(din))
        dout = torch.nn.functional.relu(self.fc2(dout))
        return torch.nn.functional.softmax(self.fc3(dout), dim=1)


train_samples, train_labels, eval_samples, eval_labels = load()
input_num, hidden_num_1, hidden_num_2, output_num = 1000, 64, 16, 2
batch_size = 500

train_set = imdbDataset(train_samples, train_labels)
test_set = imdbDataset(eval_samples, eval_labels)
train_iter = DataLoader(train_set, batch_size=batch_size)
test_iter = DataLoader(test_set, batch_size=batch_size)

model = MLP()
optimizer = torch.optim.SGD(model.parameters(),lr=0.1,momentum=0.9)  # momentum动量优化
lossfunc = torch.nn.CrossEntropyLoss()

# accuarcy
def AccuarcyCompute(pred,label):
    pred = pred.data.numpy()
    label = label.data.numpy()
    test_np = (np.argmax(pred,1) == label)
    test_np = np.float32(test_np)
    return np.mean(test_np)


for x in range(4):
    for i, data in enumerate(train_iter):
        optimizer.zero_grad()
        (inputs, labels) = data
        inputs = torch.autograd.Variable(inputs)  # 生成params?
        labels = torch.autograd.Variable(labels)  # 生成params?

        outputs = model(inputs)

        loss = lossfunc(outputs, labels)
        loss.backward()

        optimizer.step()

        if i % 100 == 0:
            print(i, ":", AccuarcyCompute(outputs, labels))