# @Author  : Edlison
# @Date    : 7/29/20 22:59
# Todo 通用类载入config
# Todo trainLoader, (evalLoader, testLoader)继承Loader
# Todo 记录参考参数avglen(sample), len(words_list)...

from math import log10
import torch
from torch.utils.data import Dataset, DataLoader


class Loader:
    """
    raw data -> train/test iter

    1. read file (path, is_test)
    2. clean words (need_cut)
    3. rm stop words (stop_word)
    4. generate words dict (dict_size, gen_dict_by)
    5. generate 'features' words to sequence (words_dict, *features_size (cut from begin))
    6. instance MyDataset (self.samples, self.labels)
    7. instance DataLoader (myDataset, batch_size)
    """

    def __init__(self):
        """
        train set need to generate dict, eval set not.
        if we need to generate dict, param dict_size is must.
        if do not need to generate dict, we need to pass param tempo_dict.

        Args:
            dict_size (): Optional
            features_size ():
            batch_size ():
            tempo_dict (): Optional
            gen_dict_by ():
            for_ ():
            rm_stop ():
        """
        self.samples = []
        self.words_dict = {}
        ...

    def get_data(self):
        """
        train loader/ eval loader return iter
        test loader return list

        Returns:

        """
        raise NotImplementedError

    def read(self):
        """
        read raw data to list.
        and whether need labels.

        Returns:

        """
        raise NotImplementedError

    # clean
    def clean(self):
        """
        zh need to cut.

        Returns:

        """
        raise NotImplementedError

    def rm_stop(self):
        """
        remove stop words.

        Returns:

        """
        for sample in self.samples:
            for index, word in enumerate(sample):
                if word in self.stop_words:
                    del sample[index]

    def load_stop_words(self, path):
        # path = '../data/stopwords/stop_en.txt'
        stop_list = []
        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                stop_list.append(line)
        self.stop_words = stop_list

    def words2seq(self, words_dict, features_size: int):
        """
        words to sequence.

        Returns:

        """
        for i, samples in enumerate(self.samples):
            if len(self.samples[i]) < features_size:
                self.samples[i] += ['<pad>'] * (features_size - len(self.samples[i]))
            else:
                # 每个样本从头部切割
                self.samples[i] = self.samples[i][:features_size]
            for j in range(features_size):
                if self.samples[i][j] not in words_dict:
                    self.samples[i][j] = words_dict['<pad>']
                else:
                    self.samples[i][j] = words_dict[self.samples[i][j]]
            # self.samples[i] = torch.tensor(self.samples[i], dtype=torch.long)

    def data2tensor(self):
        """
        pass samples, labels to tensor.

        Returns:

        """
        ...
        raise NotImplementedError


class TrainLoader(Loader):
    def __init__(self, dict_size: int, features_size: int, gen_dict_by: str, batch_size: int, rm_stop: str):
        super().__init__()
        self.labels = []
        self.batch_size = batch_size
        # 1. read
        self.read()
        print('read completed!')
        # 2. clean
        self.clean()
        print('clean completed!')
        # 3. rm stop words
        if rm_stop is not None:
            self.load_stop_words(rm_stop)
            self.rm_stop()
            print('rm stop words completed!')
        # 4. generate dict
        dict_generator = DictGenerator(self.samples, dict_size, gen_dict_by)
        self.words_dict = dict_generator.get_dict()
        print('generate dict completed!')
        # 5. words2seq
        self.words2seq(self.words_dict, features_size)
        print('words2seq completed!')
        # 6. data2tensor
        self.data2tensor()
        # 7. get data

    def get_data(self):
        return DataLoader(MyDataset(self.samples, self.labels), batch_size=self.batch_size, shuffle=True)

    def read(self):
        ...
        raise NotImplementedError

    def clean(self):
        ...
        raise NotImplementedError

    def save_dict(self, path):
        """
        must save dict.

        Args:
            path (str): the path that save words dict.

        Returns:

        """
        # path = '../data/cache/dict/dict.tempo'
        with open(path, 'w', encoding='utf-8') as f:
            f.write(str(self.words_dict))

    def data2tensor(self):
        if self.samples:
            self.samples = [torch.tensor(sample) for sample in self.samples]
        if self.labels:
            self.labels = [int(i) for i in self.labels]
            self.labels = torch.tensor(self.labels, dtype=torch.long)


class EvalLoader(Loader):
    def __init__(self, words_dict_path, features_size, batch_size: int, rm_stop: str):
        super().__init__()
        self.labels = []
        self.batch_size = batch_size
        # 1. read
        self.read()
        print('read completed!')
        # 2. clean
        self.clean()
        print('clean completed!')
        # 3. rm stop words
        if rm_stop is not None:
            self.load_stop_words(rm_stop)
            self.rm_stop()
            print('rm stop words completed!')
        # 4. load dict
        self.load_dict(words_dict_path)
        # 5. words2seq
        self.words2seq(self.words_dict, features_size)
        print('words2seq completed!')
        # 6. data2tensor
        self.data2tensor()
        # 7. get data

    def get_data(self):
        return DataLoader(MyDataset(self.samples, self.labels), batch_size=self.batch_size, shuffle=True)

    def read(self):
        ...
        raise NotImplementedError

    def clean(self):
        ...
        raise NotImplementedError

    def load_dict(self, path):
        # path = '../data/cache/dict/dict.tempo'
        with open(path, 'r', encoding='utf-8') as f:
            dict_ = eval(f.read())
        self.words_dict = dict_

    def data2tensor(self):
        if self.samples:
            self.samples = [torch.tensor(sample) for sample in self.samples]
        if self.labels:
            self.labels = [int(i) for i in self.labels]
            self.labels = torch.tensor(self.labels, dtype=torch.long)


class TestLoader(Loader):
    def __init__(self, words_dict_path, features_size, batch_size: int, rm_stop: str):
        super().__init__()
        self.batch_size = batch_size
        # 1. read
        self.read()
        print('read completed!')
        # 2. clean
        self.clean()
        print('clean completed!')
        # 3. rm stop words
        if rm_stop is not None:
            self.load_stop_words(rm_stop)
            self.rm_stop()
            print('rm stop words completed!')
        # 4. load dict
        self.load_dict(words_dict_path)
        # 5. words2seq
        self.words2seq(self.words_dict, features_size)
        print('words2seq completed!')
        # 6. data2tensor
        self.data2tensor()
        # 7. get data

    def get_data(self):
        return DataLoader(TestDataset(self.samples), batch_size=self.batch_size, shuffle=True)

    def read(self):
        ...
        raise NotImplementedError

    def clean(self):
        ...
        raise NotImplementedError

    def load_dict(self, path):
        # path = '../data/cache/dict/dict.tempo'
        with open(path, 'r', encoding='utf-8') as f:
            dict_ = eval(f.read())
        self.words_dict = dict_

    def data2tensor(self):
        if self.samples:
            self.samples = [torch.tensor(sample) for sample in self.samples]


class MyDataset(Dataset):
    def __init__(self, samples, labels):
        self.samples = samples
        self.labels = labels

    def __len__(self):
        return len(self.labels)

    def __getitem__(self, item):
        return self.samples[item], self.labels[item]


class TestDataset(Dataset):
    def __init__(self, samples):
        self.samples = samples

    def __len__(self):
        return len(self.samples)

    def __getitem__(self, item):
        return self.samples[item]


class DictGenerator:
    """
    generate dicts by (frequency, tf, idf, tf-idf).
    write into file.
    """

    def __init__(self, samples, dict_size, gen_dict_by):
        self.samples = samples
        self.dict_size = dict_size
        self.words_list = {}
        self.words_dict = {}
        self._padding_word = '<pad>'

        self.gen_words_list()
        self.tf_idf()
        self.gen_words_dict(gen_dict_by)

    def gen_words_list(self):
        words_list = {}
        for sample in self.samples:
            for word in sample:
                words_list[word] = [0, .0, .0, .0]
        self.words_list = words_list

    def tf_idf(self):
        """
        cal [frequency, tf, idf, tf-idf]

        Returns:
            self.words_list
        """
        # 计算freq
        for sample in self.samples:
            for word in sample:  # 计算term frequency 一个样本里单词的频率？？？or整个数据集中出现的频率？？？yes
                if self.words_list.get(word):
                    self.words_list[word][0] += 1
        # 计算tf
        words_num = len(self.words_list)
        for word in self.words_list:
            self.words_list[word][1] = self.words_list[word][0] / words_num
        # 计算idf
        docs_num = len(self.samples)
        for sample in self.samples:
            for word in set(sample):  # 避免一个单词在一个Sample里出现多次
                if self.words_list.get(word):
                    self.words_list[word][2] += 1.0
        for word in self.words_list:  # 计算document frequency 文档总数/单词出现过的文档数
            self.words_list[word][2] = log10(docs_num / self.words_list[word][2])
        # 计算tf-idf
        for word in self.words_list:  # 计算tf * idf 一个样本中各单词的tf_idf？？？ -> tf_idf计算应该放到循环外？？？yes
            self.words_list[word][3] = self.words_list[word][1] * self.words_list[word][2]

    def gen_words_dict(self, gen_by='frequency'):
        """
        generate words dict. use the words list which has been optimized.
        in this method, we also need to fix the dict size, if it is lager than our config.
        we can choose del words by lowest [frequency, tf, idf, tf-idf].

        Returns:
            self.words_dict

        """
        print('words list len', len(self.words_list))
        if len(self.words_list) < self.dict_size:
            self.dict_size = len(self.words_list)
        else:
            sorted_list = sorted(self.words_list.items(), key=lambda l: l[1][0], reverse=True)
            if gen_by == 'tf':
                sorted_list = sorted(self.words_list.items(), key=lambda l: l[1][1], reverse=True)
            if gen_by == 'idf':
                sorted_list = sorted(self.words_list.items(), key=lambda l: l[1][2], reverse=True)
            if gen_by == 'tf-idf':
                sorted_list = sorted(self.words_list.items(), key=lambda l: l[1][3], reverse=True)
            for i in range(self.dict_size - 1, len(self.words_list)):
                del self.words_list[sorted_list[i][0]]
        words_dict = {self._padding_word: 0}
        index = 1
        for word in self.words_list:
            if word not in words_dict:
                words_dict[word] = index
                index += 1
        self.words_dict = words_dict

    def get_dict(self):
        return self.words_dict
