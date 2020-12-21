# @Author  : Edlison
# @Date    : 12/15/20 00:39
import numpy as np
from core.v0.nn import NN

nn = NN()

X_train = np.array([[0.81, 1.02, 8.85],
                    [0.82, 0.98, 8.67],
                    [0.78, 0.99, 8.75],
                    [0.79, 1.01, 8.80],
                    [0.56, 0.85, 7.32],
                    [0.58, 0.86, 7.33],
                    [0.59, 0.83, 7.29],
                    [0.57, 0.84, 7.31]])
y_train = np.array([0,
                    0,
                    0,
                    0,
                    1,
                    1,
                    1,
                    1])
X_eval = np.array([[0.60, 0.88, 7.45],
                   [0.76, 1.00, 8.78]])

nn.fit(X_train, y_train, epochs=50000)
print(nn.predict(X_train))

print('X_train results:')
for each in X_train:
    if nn.predict(each) < 0.5:
        print(0)
    else:
        print(1)
print('X_eval results:')
for each in X_eval:
    if nn.predict(each) < 0.5:
        print(0)
    else:
        print(1)
