# @Author  : Edlison
# @Date    : 12/22/20 01:21
import numpy as np


class NeuralNetwork:
    def __init__(self, alpha=0.1):
        # 初始化权重矩阵、层数、学习率
        # 例如：layers=[2, 3, 2]，表示输入层两个结点，隐藏层3个结点，输出层2个结点
        self.W = []
        layers = [3, 4, 3, 1]
        self.layers = layers
        self.alpha = alpha
        # 随机初始化权重矩阵，如果三层网络，则有两个权重矩阵；
        # 在初始化的时候，对每一层的结点数加1，用于初始化训练偏置的权重；
        # 由于输出层不需要增加结点，因此最后一个权重矩阵需要单独初始化；
        for i in np.arange(0, len(layers)-2):
            w = np.random.randn(layers[i] + 1, layers[i + 1] + 1)
            self.W.append(w / np.sqrt(layers[i]))
        # 初始化最后一个权重矩阵
        w = np.random.randn(layers[-2] + 1, layers[-1])
        self.W.append(w / np.sqrt(layers[-2]))

    def __repr__(self):
        # 输出网络结构
        return "NeuralNetwork: {}".format(
            "-".join(str(l) for l in self.layers)
        )

    def sigmoid(self, x):
        # sigmoid激活函数
        return 1.0 / (1 + np.exp(-x))

    def sigmoid_deriv(self, x):
        # sigmoid的导数
        return x * (1 - x)

    def fit(self, X, y, epochs=1000, display=100):
        # 训练网络
        # 对训练数据添加一维值为1的特征，用于同时训练偏置的权重
        X = np.c_[X, np.ones(X.shape[0])]
        # 迭代的epoch
        for epoch in np.arange(0, epochs):
            # 对数据集中每一个样本执行前向传播、反向传播、更新权重
            for (x, target) in zip(X, y):
                self.fit_partial(x, target)
            # 打印输出
            if epoch == 0 or (epoch + 1) % display == 0:
                loss = self.calculate_loss(X, y)
                print("epoch={}, loss={:.7f}".format(
                    epoch + 1, loss
                ))

    def fit_partial(self, x, y):
        # 构造一个列表A，用于保存网络的每一层的输出，即经过激活函数的输出
        A = [np.atleast_2d(x)]

        # ---------- 前向传播 ----------
        # 对网络的每一层进行循环
        for layer in np.arange(0, len(self.W)):
            # 计算当前层的输出
            net = A[layer].dot(self.W[layer])
            out = self.sigmoid(net)
            # 添加到列表A
            A.append(out)

        # ---------- 反向传播 ----------
        # 计算error
        error = A[-1] - y
        # 计算最后一个权重矩阵的D[?]
        D = [error * self.sigmoid_deriv(A[-1])]
        # 计算前面的权重矩阵的D[?]
        for layer in np.arange(len(A)-2, 0, -1):
            # 参见上文推导的公式
            delta = D[-1].dot(self.W[layer].T)
            delta = delta * self.sigmoid_deriv(A[layer])
            D.append(delta)
        # 列表D是从后往前记录，下面更新权重矩阵的时候，是从输入层到输出层
        #　因此，在这里逆序
        D = D[::-1]
        # 迭代更新权重
        for layer in np.arange(0, len(self.W)):
            # 参考上文公式
            self.W[layer] += -self.alpha * A[layer].T.dot(D[layer])

    def predict(self, X, addBias=True):
        # 预测
        p = np.atleast_2d(X)
        # check to see if the bias column should be added
        if addBias:
            # insert a column of 1's as the last entry in the feature
            # matrix (bias)
            p = np.c_[p, np.ones((p.shape[0]))]
        # loop over our layers int the network
        for layer in np.arange(0, len(self.W)):
            # computing the output prediction is as simple as taking
            # the dot product between the current activation value 'p'
            # and the weight matrix associated wieth the current layer,
            # then passing this value through a nonlinear activation
            # function
            p = self.sigmoid(np.dot(p, self.W[layer]))
        # return the predicted value
        return p

    def calculate_loss(self, X, targets):
        # make predictions for the input data points then compute
        # the loss
        targets = np.atleast_2d(targets)
        predictions = self.predict(X, addBias=False)
        loss = 0.5 * np.sum((predictions - targets) ** 2)
        # return the loss
        return loss

if __name__ == '__main__':
    nn = NeuralNetwork([3, 4, 3, 1], alpha=0.01)
    print(nn.__repr__())
    X_train_ = np.array([[0, 0],
                         [1, 0],
                         [0, 1],
                         [1, 1]])
    y_train_ = np.array([0,
                         1,
                         1,
                         0])

    X_ = np.array([[0.81, 1.02, 8.85],
                   [0.82, 0.98, 8.67],
                   [0.78, 0.99, 8.75],
                   [0.79, 1.01, 8.80],
                   [0.56, 0.85, 7.32],
                   [0.58, 0.86, 7.33],
                   [0.59, 0.83, 7.29],
                   [0.57, 0.84, 7.31]])
    y_ = np.array([0,
                   0,
                   0,
                   0,
                   1,
                   1,
                   1,
                   1])
    X_eval = np.array([[0.60, 0.88, 7.45],
                      [0.76, 1.00, 8.78]])

    nn.fit(X_, y_, epochs=50000)
    print(nn.predict(X_))

    for each in X_:
        if nn.predict(each) < 0.5:
            print(0)
        else:
            print(1)

    for each in X_eval:
        if nn.predict(each) < 0.5:
            print(0)
        else:
            print(1)