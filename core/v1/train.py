# @Author  : Edlison
# @Date    : 7/30/20 03:16
# TODO Trainer训练器与Tester测试器分开
# TODO 导出模型 导入模型

import torch
from core.v1.model import Model


class Trainer:
    """
    import Model, epoch.
    no need implement.
    """
    def __init__(self, train_iter, eval_iter, model:Model, criterion, optimizer,epoch):
        self.train_iter = train_iter
        self.eval_iter = eval_iter
        self.model = model
        self.criterion = criterion
        self.optimizer = optimizer
        self.epoch = epoch

    def eval(self, iter):
        correct, loss_sum, num = 0, 0.0, 0
        with torch.no_grad():
            for X, y in iter:
                outputs = self.model(X)
                outputs_ = torch.argmax(outputs, dim=-1)
                loss = self.criterion(outputs, y)
                loss_sum += loss
                correct += torch.sum(outputs_.eq(y)).float().item()
                num += y.size()[0]
        print(f'loss {loss_sum/num:.5f}, acc {correct/num:.5f}.')

    def start(self):
        for i in range(self.epoch):
            self.model.train()  # 训练模式
            print(f'epoch {i} start')
            for j, data in enumerate(self.train_iter):
                (X, labels) = data
                # print('before model X', X, X.shape)  # [500, 30], [500]
                outputs = self.model(X)
                # print('after model output', outputs, outputs.shape)  # [30, 2] -> [500, 2]
                # print('after model labels', labels, labels.shape)  # [500]
                loss = self.criterion(outputs, labels)
                self.optimizer.zero_grad()
                loss.backward()
                self.optimizer.step()
            print('train set', end=' ')
            self.eval(self.train_iter)
            print('eval set', end=' ')
            self.model.eval()  # 评价模式 加上train set与eval set的acc一致？？？
            self.eval(self.eval_iter)

    def save_model(self, path):
        """
        must export model, if we need generate test labels.

        Args:
            path (str):

        Returns:

        """
        # path = '../data/cache/model/imdb.pt'
        torch.save(self.model, path)
