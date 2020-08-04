from core.processor import wordsbag
import json
import time
from sklearn.linear_model.logistic import LogisticRegression
from sklearn.metrics import accuracy_score


def predict_imdb_test():
    p.imdb()
    p.load_en_test()
    p.split_test_words()

    train_pow = p.pow_tf(p.words_dict, p.train_X)
    test_pow = p.pow_tf(p.words_dict, p.test_X)

    clf = LogisticRegression()
    clf.fit(train_pow, p.train_y)
    pre = clf.predict(test_pow)
    with open('../data/output/imdb_out_0.txt', 'w') as f:
        for i in pre:
            f.write(i + '\n')


def predict_imdb_eval():
    p.imdb()
    p.load_en_eval()
    p.split_eval_words()

    train_pow = p.pow_tf_idf(p.words_dict, p.train_X)
    eval_pow = p.pow_tf_idf(p.words_dict, p.eval_X)

    clf = LogisticRegression()
    clf.fit(train_pow, p.train_y)
    pre = clf.predict(eval_pow)

    score = accuracy_score(p.eval_y, pre)
    print(score)
    # freq 0.952
    # tf 0.953
    # tf-idf 0.772

def predict_cnews_test():
    time_0 = time.time()
    p.cnews()
    p.load_zh_test()
    p.split_zh_test()
    time_1 = time.time()
    train_pow = p.pow_boolean(p.words_dict, p.train_X)
    test_pow = p.pow_boolean(p.words_dict, p.test_X)
    time_2 = time.time()
    clf = LogisticRegression()
    clf.fit(train_pow, p.train_y)
    pre = clf.predict(test_pow)
    time_3 = time.time()
    print('gen dict and load eval', time_1 - time_0)
    print('gen pow', time_2 - time_1)
    print('train model', time_3 - time_2)
    print('total', time_3 - time_0)
    with open('../data/output/cnews_out.txt', 'w') as f:
        for i in pre:
            f.write(p.tags_rev[i] + '\n')
    # 0.9362


if __name__ == '__main__':
    cnews_configdir = '../core/config/cnews_config.json'
    imdb_configdir = '../core/config/imdb_config.json'

    with open(imdb_configdir, 'r') as f:
        config = json.load(f)

    p = wordsbag.DataProcessor(config)

    predict_imdb_test()