#!/usr/bin/env python2
# -*- coding: utf-8 -*-
import re
import logging
import numpy as np
import pandas as pd
from nltk.corpus import stopwords
from nltk.tokenize import word_tokenize
from nltk.stem import WordNetLemmatizer
from gensim.models.word2vec import Word2Vec

wordnet_lemmatizer = WordNetLemmatizer()  # 词性还原
stop_word = stopwords.words('english')


def hasNumber(word):
    return bool(re.match(r'/d', word))


def isSymbol(word):
    return bool(re.match(r'[^\w]', word))


def check(word):
    if word in stop_word:
        return False
    elif hasNumber(word) or isSymbol(word):
        return False
    else:
        return True


def clear_data(sentence):
    ret = []
    for word in sentence:
        if check(word):
            word = word.lower().replace('b"', '').replace("b'", '').replace('"', '').replace("'", '')
            ret.append(wordnet_lemmatizer.lemmatize(word))  # 词性还原
    return ret


# 取一句话每个词向量相加的平均值
def get_vector(word_list):
    res = np.zeros([128])
    count = 0
    for word in word_list:
        if word in vocab:
            res += model[word]
            count += 1
    return res/count


# 去前128个字作，使每句话为一个128*128的矩阵
def build_matrix(data, row=128, columns=128):
    ret = []
    for item in data:
        matrix = []
        for i in range(row):
            try:
                matrix.append(model[item[i]].tolist())
            except:
                matrix.append([0]*columns)
        ret.append(matrix)
    return ret


if __name__ == '__main__':
    data = pd.read_csv('data/Combined_News_DJIA.csv')
    train = data[data['Date'] < '2015-01-01']
    test = data[data['Date'] > '2014-12-31']

    # 数据向量

    X_train = train[train.columns[2:]]  # 取数据
    combine = X_train.values.flatten().astype(str)  # 将每个行25个句子合并

    X_train = X_train.values.astype(str)
    X_train = np.array([' '.join(sentence) for sentence in X_train])

    X_test = test[test.columns[2:]]
    X_test = np.array([' '.join(sentence) for sentence in X_test])
    # 类标签
    Y_train = train[train.columns[1]].values
    Y_test = test[test.columns[1]].values


    combine = [word_tokenize(sentence) for sentence in combine]  # 用来训练
    X_train = [word_tokenize(sentence) for sentence in X_train]  # 用来将数据转化为矩阵，方便CNN训练
    X_test = [word_tokenize(sentence) for sentence in X_test]

    combine = [clear_data(sentence) for sentence in combine]
    X_train = [clear_data(sentence) for sentence in X_train]
    X_test = [clear_data(sentence) for sentence in X_test]
    logging.basicConfig(format='%(asctime)s : %(levelname)s : %(message)s', level=logging.INFO)
    model = Word2Vec(combine, size=128, window=5, min_count=5, workers=4)
    vocab = model.vocab

    print(get_vector(['At', 'that', 'moment', 'when', 'it', 'has', 'the', 'capability', 'to', 'hit', 'the', 'U.S']))
    # 此时X_train可以作为CNN输入向量了
    X_train = build_matrix(X_train)
    Y_train = build_matrix(Y_train)












