# @Author  : Edlison
# @Date    : 8/4/20 00:46
import torch
import numpy as np


class Tester:
    """
    tester
    Params
    need model, test_iter
    Methods

    """
    def __init__(self, test_iter, model_path):
        self.test_iter = test_iter
        self.model = torch.load(model_path)

    def test(self):
        self.model.eval()
        with torch.no_grad():
            res = []
            for X in self.test_iter:
                outputs = self.model(X)
                res += torch.argmax(outputs, -1)
                # temp = torch.argmax(outputs, -1)
                # temp = temp.numpy()
                # res.extend(temp)  # pytorch版本问题 不能直接加tensor 需要转换成numpy
        self.test_label = res
        print('labels len:', len(res))

    def export(self, path):
        """
        need to override.

        Args:
            path (str): save labels to.

        Returns:

        """
        with open(path, 'w', encoding='utf-8') as f:
            for i in self.test_label:
                f.write(str(int(i)) + '\n')
