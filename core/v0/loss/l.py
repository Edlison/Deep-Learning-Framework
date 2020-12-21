# @Author  : Edlison
# @Date    : 12/15/20 00:49
"""
损失函数
包括反向传播 继承
"""
import numpy as np


class CrossEntropyLoss:
    def func(self, out_, y_):
        out_ = np.array(out_)
        y_ = np.array(y_)
        return 1 / 2 * (out_ - y_) ** 2

    def derivative(self, out_, y_):
        out_ = np.array(out_)
        y_ = np.array(y_)
        return out_ - y_