# @Author  : Edlison
# @Date    : 7/25/20 19:30
import torch as t
import numpy as np
import random
from core.processor.wordsbag import DataProcessor
import json
from torch.utils.data import Dataset, DataLoader

num_input = 1000
num_output = 2

W = t.tensor(np.random.normal(0, 0.1, (num_input, num_output)), dtype=t.float)
b = t.zeros(num_output, dtype=t.float)

W.requires_grad_()
b.requires_grad_()


# 将数据遍历， 每次返回一个batch的数据
def data_iter(batch_size, features, labels):
    num_examples = len(features)
    indices = list(range(num_examples))
    random.shuffle(indices)  # 样本的读取顺序是随机的
    for i in range(0, num_examples, batch_size):
        j = t.LongTensor(indices[i: min(i + batch_size, num_examples)])  # 最后一次可能不足一个batch
        yield features.index_select(0, j), labels.index_select(0, j)


# softMax函数
def softmax(X):
    X_exp = X.exp()
    partition = X_exp.sum(dim=1, keepdim=True)
    return X_exp / partition


# 模型
def net(X):
    return softmax(t.mm(X.view(-1, num_input).float(), W) + b)  # X需要Float()!!!


# 交叉熵损失函数
def cross_entropy(y_hat, y):
    return - t.log(y_hat.gather(1, y.view(-1, 1)))


def evaluate_accuracy(data_iter, net):
    acc_sum, n = 0.0, 0
    for X, y in data_iter:
        acc_sum += (net(X).argmax(dim=1) == y).float().sum().item()
        n += y.shape[0]
    return acc_sum / n


def sgd(params, lr, batch_size):  # 本函数已保存在d2lzh_pytorch包中方便以后使用
    for param in params:
        param.data -= lr * param.grad / batch_size  # 注意这里更改param时用的param.data


num_inputs, num_outputs, num_hiddens_1, num_hiddens_2 = 1000, 2, 64, 16

bed_W1 = t.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens_1)), dtype=t.float)
bed_b1 = t.zeros(num_hiddens_1, dtype=t.float)
bed_W2 = t.tensor(np.random.normal(0, 0.01, (num_hiddens_1, num_hiddens_2)), dtype=t.float)
bed_b2 = t.zeros(num_hiddens_2, dtype=t.float)
bed_W3 = t.tensor(np.random.normal(0, 0.01, (num_hiddens_2, num_outputs)), dtype=t.float)
bed_b3 = t.zeros(num_outputs, dtype=t.float)

mlp_W1 = t.tensor(np.random.normal(0, 0.01, (num_inputs, num_hiddens_1)), dtype=t.float)
mlp_b1 = t.zeros(num_hiddens_1, dtype=t.float)
mlp_W2 = t.tensor(np.random.normal(0, 0.01, (num_hiddens_1, num_outputs)), dtype=t.float)
mlp_b2 = t.zeros(num_outputs, dtype=t.float)

bed_params = [bed_W1, bed_b1, bed_W2, bed_b2, bed_W3, bed_b3]
for param in bed_params:
    param.requires_grad_()

mlp_params = [mlp_W1, mlp_b1, mlp_W2, mlp_b2]
for param in mlp_params:
    param.requires_grad_()


def relu(X):
    return t.max(X, other=t.tensor(0.0))
# def net():
# 只使用nn.CrossEntropy 有一层softmax lr=[1000, 100] acc=[0.861, 0.803]
# 使用cross_entropy lr=0.1 acc=0.815
def net_mlp(X):
    # 使用nn.CrossEntropy 最后一层softmax lr=1000 acc=0.859 | 最后一层relu lr=[1000, 100] acc=[0.5, 0.5] | 最后一层没有激活函数 lr=[1000, 100] acc=[0.836, 0.5]
    # 使用cross_entropy loss一直为nan
    X = X.view((-1, num_inputs)).float()
    Layer1 = relu(t.matmul(X, mlp_W1) + mlp_b1)
    return t.matmul(Layer1, mlp_W2) + mlp_b2  # 最后一层没有激活函数 什么意思???


def net_bed(X):
    # 使用nn.CrossEntropy lr=1000 0.836 | lr=100 0.500
    # 使用cross_entropy lr=100 loss一直为nan | lr=1 0.816
    X = X.view((-1, num_inputs)).float()
    Layer1 = relu(t.matmul(X, bed_W1) + bed_b1)
    Layer2 = relu(t.matmul(Layer1, bed_W2) + bed_b2)
    return softmax(t.matmul(Layer2, bed_W3) + bed_b3)


batch_size = 500
num_epochs = 5
lr = 100


def train_ch3(net, train_iter, test_iter, loss, num_epochs, batch_size,
              params=None, lr=None, optimizer=None):
    for epoch in range(num_epochs):
        train_l_sum, train_acc_sum, n = 0.0, 0.0, 0
        for X, y in train_iter:
            y_hat = net(X)
            l = loss(y_hat, y).sum()

            # 梯度清零
            if optimizer is not None:
                optimizer.zero_grad()
            elif params is not None and params[0].grad is not None:
                for param in params:
                    param.grad.data.zero_()

            l.backward()

            if optimizer is None:
                sgd(params, lr, batch_size)
            else:
                optimizer.step()

            train_l_sum += l.item()  # loss值每个batch的相加？？？
            train_acc_sum += (y_hat.argmax(dim=1) == y).sum().item()  # y_hat预测值与真实值相等时 计数
            n += y.shape[0]  # 一个batch总个数
        test_acc = evaluate_accuracy(test_iter, net)  # 训练一个epoch后 带入模型评价准确度
        print('epoch %d, loss %.4f, train acc %.3f, test acc %.3f'
              % (epoch + 1, train_l_sum / n, train_acc_sum / n, test_acc))


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
train_samples = t.tensor(dp.pow_boolean(dp.words_dict, dp.train_X))
label_0 = [int(_) for _ in dp.train_y]
train_labels = t.tensor(label_0)

eval_samples = t.tensor(dp.pow_boolean(dp.words_dict, dp.eval_X))
label_1 = [int(_) for _ in dp.eval_y]
eval_labels = t.tensor(label_1)
print('start iter')


class MLP(t.nn.Module):
    def __init__(self):
        super(MLP, self).__init__()
        self.fc1 = t.nn.Linear(num_inputs, num_hiddens_1)
        self.fc2 = t.nn.Linear(num_hiddens_1, num_hiddens_2)
        self.fc3 = t.nn.Linear(num_hiddens_2, num_outputs)

    def forward(self, din):
        din = din.view(-1, num_inputs)
        dout = t.nn.functional.relu(self.fc1(din))
        dout = t.nn.functional.relu(self.fc2(dout))
        return t.nn.functional.softmax(self.fc3(dout))


# 自己定义的Dataset继承Dataset 注意重写其中的len, getitem, 2个成员方法
# DataLoader需要重写1个成员方法 iter(self)把自己装进去
# 将自己的Dataset传入DataLoader
class imdbDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.samples[item], self.labels[item]


train_set = imdbDataset(train_samples, train_labels)
test_set = imdbDataset(eval_samples, eval_labels)
train_iter = DataLoader(train_set, batch_size=batch_size)
test_iter = DataLoader(test_set, batch_size=batch_size)

model = MLP()
model.parameters()

print('start train')
train_ch3(net_bed, train_iter, test_iter, cross_entropy, num_epochs, batch_size, bed_params, lr, optimizer=None)



# ？？？
# 最后一层输出不加激活函数 含义？？？
