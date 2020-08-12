# @Author  : Edlison
# @Date    : 7/30/20 03:16

import torch
from core.v1.model import Model
import tqdm
import time


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

    def _train(self, model, iter, optimizer, criterion):
        epoch_loss, epoch_acc, total_len = 0, 0, 0
        model.train()

        for X, y in iter:
            # 梯度清零
            optimizer.zero_grad()
            # 送入模型
            outputs = model(X)
            # 损失函数
            loss = criterion(outputs, y)
            # 准确率
            acc = self._acc_num(outputs, y)
            # 反向传播
            loss.backward()
            # 梯度下降
            optimizer.step()

            epoch_loss += loss
            epoch_acc += acc
            total_len += len(y)
        print(total_len)
        return epoch_loss / total_len, epoch_acc / total_len

    def _eval(self, model, iter, criterion):
        epoch_loss, epoch_acc, total_len = 0, 0, 0
        model.eval()

        with torch.no_grad():
            # 没有反向传播和梯度下降
            for X, y in iter:
                outputs = model(X)
                loss = criterion(outputs, y)
                acc = self._acc_num(outputs, y)

                epoch_loss += loss
                epoch_acc += acc
                total_len += len(y)

        return epoch_loss / total_len, epoch_acc / total_len

    def start(self):
        start = time.time()
        for epoch in tqdm.trange(self.epoch):
            epoch_start = time.time()
            train_loss, train_acc = self._train(self.model, self.train_iter, self.optimizer, self.criterion)
            eval_loss, eval_acc = self._eval(self.model, self.eval_iter, self.criterion)
            epoch_end = time.time()

            print(f'\nEpoch: {epoch + 1:02} | Epoch Time: {epoch_end - epoch_start:.2f}s')
            print(f'\tTrain Loss: {train_loss:.4f} | Train Acc: {train_acc:.4f}')
            print(f'\tEval. Loss: {eval_loss:.4f} | Eval. Acc: {eval_acc:.4f}')
        end = time.time()
        print(f'\nTotal Train Time: {end - start:.2f}s')

    def _acc_num(self, prediction, true):
        """
        dim = 1

        Args:
            prediction ():
            true ():

        Returns:

        """
        prediction = torch.argmax(prediction, dim=1)
        acc = torch.sum(prediction.eq(true)).float().item()
        return acc / len(true)

    def save_model(self, path):
        """
        must export model, if we need generate test labels.

        Args:
            path (str):

        Returns:

        """
        torch.save(self.model, path)

    def eval_test(self, iter):  # withdraw
        res = []
        self.model.eval()
        with torch.no_grad():
            for X, y in iter:
                outputs = self.model(X)
                outputs = torch.argmax(outputs, dim=-1)
                res.extend(outputs)

        with open('../data/output/imdb_v1_out_2.txt', 'w', encoding='utf-8') as f:
            for i in res:
                f.write(str(int(i)) + '\n')