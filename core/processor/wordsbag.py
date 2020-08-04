from math import log10
import time
import string
import re
# import jieba


class DataProcessor:
    def __init__(self, config):
        """
        init class.

        Args:
            config (map): config file
        """

        model_obj = "model"
        dict_size_obj = "dict_size"
        split_sign_obj = "split_sign"
        train_data_path_obj = "train_data_path"
        eval_data_path_obj = "eval_data_path"
        test_data_path_obj = "test_data_path"
        stop_words_path_obj = "stop_word_path"
        output_path_obj = "output_path"

        if config[dict_size_obj]:
            self.dict_size = int(config[dict_size_obj])
        else:
            self.dict_size = None

        if config[split_sign_obj]:
            self.split_sign = config[split_sign_obj]
        else:
            self.split_sign = None

        if config[train_data_path_obj]:
            self.train_data_path = config[train_data_path_obj]
        else:
            self.train_data_path = None

        if config[eval_data_path_obj]:
            self.eval_data_path = config[eval_data_path_obj]
        else:
            self.eval_data_path = None

        if config[test_data_path_obj]:
            self.test_data_path = config[test_data_path_obj]
        else:
            self.test_data_path = None

        if config[stop_words_path_obj]:
            self.stop_words_path = config[stop_words_path_obj]
        else:
            self.stop_words_path = None

        if config[output_path_obj]:
            self.output_path = config[output_path_obj]
        else:
            self.output_path = None

    def load_zh(self, set='train'):
        """
        load raw data from chinese.

        Args:
            set (str): train set or eval set

        Returns:
            self.train_X or self.eval_X
            self.train_y or self.eval_y
            (tages)

        """
        context = []
        labels = []
        labels_num = []
        tags = {}
        tags_rev = {}

        # 通过参数判断处理的数据集
        path = self.train_data_path
        if set == 'eval':
            path = self.eval_data_path

        with open(path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                labels.append(line.split(self.split_sign, 1)[0].strip())
                context.append(line.split(self.split_sign, 1)[1].strip())
        index = 0
        for item in labels:
            if item not in tags:
                tags[item] = index
                index += 1
        for item in labels:
            labels_num.append(tags[item])
        for item in tags:
            tags_rev[tags[item]] = item

        # 通过参数判断处理的数据集
        if set == 'train':
            self.train_X = context
            self.train_y = labels_num
            self.tags = tags
            self.tags_rev = tags_rev
        if set == 'eval':
            self.eval_X = context
            self.eval_y = labels_num
            self.tags = tags
            self.tags_rev = tags_rev

    def load_zh_test(self):
        """
        only load test set.

        Returns:
            self.train_X or self.eval_X

        """
        context = []
        with open(self.test_data_path, 'r', encoding='utf-8') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                context.append(line.strip())
        self.test_X = context

    def split_zh(self, set='train'):
        """
        split

        Args:
            set ():

        Returns:

        """
        pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  # 只保留中英文、数字和符号，去掉其他东西
        # 若只保留中英文和数字，则替换为[^\u4e00-\u9fa5^a-z^A-Z^0-9]
        data = []

        # 通过参数判断处理的数据集
        context = self.train_X
        if set == 'eval':
            context = self.eval_X

        for line in context:
            line = re.sub(pattern, '', line)  # 把文本中匹配到的字符替换成空字符
            line = ''.join(line.split())  # 去除空白
            # line = list(jieba.cut(line))  # lcut
            data.append(line)

        # 通过参数判断处理的数据集
        if set == 'train':
            self.train_X = data
        if set == 'eval':
            self.eval_X = data

    def split_zh_test(self):
        pattern = re.compile("[^\u4e00-\u9fa5^a-z^A-Z^0-9]")  # 只保留中英文、数字和符号，去掉其他东西
        # 若只保留中英文和数字，则替换为[^\u4e00-\u9fa5^a-z^A-Z^0-9]
        data = []
        for line in self.test_X:
            line = re.sub(pattern, '', line)  # 把文本中匹配到的字符替换成空字符
            line = ''.join(line.split())  # 去除空白
            # line = list(jieba.cut(line))
            data.append(line)
        self.test_X = data

    def load_en_train(self):
        """
        read raw train data, generate train_X, train_y.

        Returns:
            self.train_X
            self.train_y

        """
        context = []
        labels = []
        with open(self.train_data_path, 'r') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                context.append(line.split(self.split_sign)[0].strip().lower())  # 去除前后空格 文本转为小写
                labels.append(line.split(self.split_sign)[1].strip().lower())
        self.train_X = context
        self.train_y = labels

    def load_en_eval(self):
        """
        read raw eval data and split sample into eval_X, eval_y.

        Returns:
            self.eval_X
            self.eval_y

        """
        context = []
        labels = []
        with open(self.eval_data_path, 'r') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                context.append(line.split(self.split_sign)[0].strip().lower())  # 去除前后空格 文本转为小写
                labels.append(line.split(self.split_sign)[1].strip().lower())
        self.eval_X = context
        self.eval_y = labels

    def load_en_test(self):
        """
        read raw test data and split sample into test_X, test_y.

        Returns:
            self.eval_X
            self.eval_y

        """
        context = []
        with open(self.test_data_path, 'r') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                context.append(line.split(self.split_sign)[0].strip().lower())  # 去除前后空格 文本转为小写
        self.test_X = context

    def split_train_words(self):  # 移除标点符号
        """
        read train_X and split train_X into words.

        Returns:
            self.train_X

        """
        data = []
        for sample in self.train_X:
            rm_char = string.punctuation  # 拿到全部英文标点符号和(数字?)
            for each in rm_char:  # 移除英文标点符号
                sample = sample.replace(each, '')
            sample = sample.split(' ')
            data.append(sample)
        self.train_X = data

    def split_eval_words(self):  # 移除标点符号
        """
        read eval_X and split eval_X into words.

        Returns:
            self.eval_X

        """
        data = []
        for sample in self.eval_X:
            rm_char = string.punctuation  # 拿到全部英文标点符号和(数字?)
            for each in rm_char:  # 移除英文标点符号
                sample = sample.replace(each, '')
            sample = sample.split(' ')
            data.append(sample)
        self.eval_X = data

    def split_test_words(self):  # 移除标点符号
        """
        read test_X and split test_X into words.

        Returns:
            self.test_X

        """
        data = []
        for sample in self.test_X:
            rm_char = string.punctuation  # 拿到全部英文标点符号和(数字?)
            for each in rm_char:  # 移除英文标点符号
                sample = sample.replace(each, '')
            sample = sample.split(' ')
            data.append(sample)
        self.test_X = data

    def gen_words_dict(self, rm_by='frequency'):
        """
        generate words dict. use the words list which has been optimized.
        in this method, we also need to fix the dict size, if it is lager than our config.
        we can choose del words by lowest [frequency, tf, idf, tf-idf]

        Returns:
            self.words_dict

        """
        if len(self.words_list) < self.dict_size:
            self.dict_size = len(self.words_list)
        else:
            sorted_list = sorted(self.words_list.items(), key=lambda l: l[1][0], reverse=True)
            if rm_by == 'tf':
                sorted_list = sorted(self.words_list.items(), key=lambda l: l[1][1], reverse=True)
            if rm_by == 'idf':
                sorted_list = sorted(self.words_list.items(), key=lambda l: l[1][2], reverse=True)
            if rm_by == 'tf-idf':
                sorted_list = sorted(self.words_list.items(), key=lambda l: l[1][3], reverse=True)
            for i in range(self.dict_size, len(self.words_list)):
                del self.words_list[sorted_list[i][0]]
        words_dict = {}
        index = 0
        for word in self.words_list:
            if word not in words_dict:
                words_dict[word] = index
                index += 1
        self.words_dict = words_dict

    def load_stop_words(self):
        """
        load stop words.

        Returns:
            self.stop_words

        """
        stop_list = []
        with open(self.stop_words_path, 'r') as f:
            for line in f.readlines():
                line = line.split('\n')[0]
                stop_list.append(line)
        self.stop_words = stop_list

    # 去除停词 + 去除低频词
    def rm_stop_and_gen_words_list(self):
        """
        rm stop words.
        init words list, which has 4 dimensions {word: [frequency, tf, idt, tf-idf]}.

        Args:
            threshold (int): rm word, if word frequency below this threshold. (removed)

        Returns:
            self.words_list

        """
        words_list = {}
        for sample in self.train_X:  # 去除停词
            for word in sample:
                if word not in self.stop_words:
                    words_list[word] = [0, .0, .0, .0]  # 生成词列表
        self.words_list = words_list

    def tf_idf(self):
        """
        cal [frequency, tf, idf, tf-idf]

        Returns:
            self.words_list
        """
        # 计算freq
        for sample in self.train_X:
            for word in sample:  # 计算term frequency 一个样本里单词的频率？？？or整个数据集中出现的频率？？？yes
                if self.words_list.get(word):
                    self.words_list[word][0] += 1
        # 计算tf
        words_num = len(self.words_list)
        for word in self.words_list:
            self.words_list[word][1] = self.words_list[word][0] / words_num
        # 计算idf
        docs_num = len(self.train_X)
        for sample in self.train_X:
            for word in set(sample):  # 避免一个单词在一个Sample里出现多次
                if self.words_list.get(word):
                    self.words_list[word][2] += 1.0
        for word in self.words_list:  # 计算document frequency 文档总数/单词出现过的文档数
            self.words_list[word][2] = log10(docs_num / self.words_list[word][2])
        # 计算tf-idf
        for word in self.words_list:  # 计算tf * idf 一个样本中各单词的tf_idf？？？ -> tf_idf计算应该放到循环外？？？yes
            self.words_list[word][3] = self.words_list[word][1] * self.words_list[word][2]

    def imdb(self):
        self.load_en_train()
        self.load_stop_words()
        self.split_train_words()
        self.rm_stop_and_gen_words_list()
        self.tf_idf()
        self.gen_words_dict()

    def cnews(self):
        self.load_zh()
        self.split_zh()
        self.load_stop_words()
        self.rm_stop_and_gen_words_list()
        self.tf_idf()
        self.gen_words_dict()

    def pow_boolean(self, words_dict, data):  # 词袋模型-0/1
        res = []

        for sample in data:
            sample2vector = [0 for _ in range(len(words_dict))]
            for word in sample:
                if word in words_dict:
                    sample2vector[words_dict[word]] = 1
            res.append(sample2vector)

        return res

    def pow_tf(self, words_dict, data):  # 词袋模型-tf
        res = []

        for sample in data:
            sample2vector = [0 for _ in range(len(words_dict))]
            for word in sample:
                if word in words_dict:
                    sample2vector[words_dict[word]] += 1
            res.append(sample2vector)

        return res

    def pow_tf_idf(self, words_dict, data):
        res = []

        for sample in data:
            sample2vector = [0 for _ in range(len(words_dict))]
            for word in sample:
                if (word in words_dict) and (word in self.words_list):
                    sample2vector[words_dict[word]] = self.words_list[word][3]  # tf并不是一个sample的 这里取的是corpus的
            res.append(sample2vector)
        return res

# Need To Do !!!
# 继承base load(选择 train, eval, test) split(选择 train, eval, test)
