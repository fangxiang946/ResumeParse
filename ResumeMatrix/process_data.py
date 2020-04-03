import os
import re
import sys
import json
import pickle
import logging
import itertools
import numpy as np
import pandas as pd
import gensim as gs
import jieba
from pprint import pprint
from collections import Counter
from tensorflow.contrib import learn
import gensim
from sklearn.model_selection import train_test_split

logging.getLogger().setLevel(logging.INFO)

pattern = re.compile(r'(\d)')


def clean_str(s):
    s = s.replace('？', '?') \
        .replace('。', ' . ') \
        .replace('，', ',') \
        .replace('；', ' ; ') \
        .replace('：', ':') \
        .replace('【', '[') \
        .replace('】', ']') \
        .replace('￥', '$') \
        .replace('……', '^') \
        .replace('、', ',') \
        .replace('‘', "'") \
        .replace('’', "'") \
        .replace('“', '"') \
        .replace('”', '"') \
        .replace('（', '(') \
        .replace('）', ')')
    s = re.sub(r"[^\u4e00-\u9fa5\-\.\/\@\[A-Za-z0-9:(),!?\'\`]", " ", s)
    s = re.sub(r" : ", ":", s)
    s = re.sub(r"\'s", " \'s", s)
    s = re.sub(r"\'ve", " \'ve", s)
    s = re.sub(r"n\'t", " n\'t", s)
    s = re.sub(r"\'re", " \'re", s)
    s = re.sub(r"\'d", " \'d", s)
    s = re.sub(r"\'ll", " \'ll", s)
    s = re.sub(r",", " , ", s)
    s = re.sub(r"!", " ! ", s)
    s = re.sub(r"\(", " \( ", s)
    s = re.sub(r"\)", " \) ", s)
    s = re.sub(r"\[", " \[ ", s)
    s = re.sub(r"\]", " \] ", s)
    s = re.sub(r"\?", " \? ", s)
    s = re.sub(r"\s{2,}", " ", s)
    words = jieba.lcut(s.strip().lower(), HMM=False)
    result = []
    for i in range(len(words)):
        word = words[i]
        list = re.split(pattern, word)
        list = [item for item in filter(lambda x: x != '', list)]
        result = result + list
    return result


def pad_sentences(sentences, padding_word='<PAD/>', forced_sequence_length=None):
    """pad sentences during training or prediction"""
    if forced_sequence_length is None:
        sequence_length = max(len(x) for x in sentences)
    else:
        logging.critical('this is prediction ,readinig the trained sequence length')
        sequence_length = forced_sequence_length
    logging.critical('the maximun length is {}'.format(sequence_length))

    padded_sentences = []
    for i in range(len(sentences)):
        sentence = sentences[i]
        num_padding = sequence_length - len(sentence)

        if num_padding < 0:
            padded_sentence = sentence[0:sequence_length]
            logging.info('"%s" has to be cut off because it is longer than max_len ' % (' '.join(padded_sentence)))
        else:
            padded_sentence = sentence + [padding_word] * num_padding
        padded_sentences.append(padded_sentence)
    return padded_sentences


def load_embeddings(vocabulary, word2vec_path=None):
    word_embeddings = {}
    if word2vec_path is not None:
        word2vec = gensim.models.Word2Vec.load(word2vec_path)
    for word in vocabulary:
        if word2vec_path is not None and word in word2vec.wv.vocab:
            word_embeddings[word] = word2vec.wv[word]
        else:
            word_embeddings[word] = np.random.uniform(-0.25, 0.25, 256)
    del word2vec
    return word_embeddings


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    data = np.array(data)
    data_size = len(data)
    num_batches_per_epoch = int(data_size / batch_size) + 1

    for epoch in range(num_epochs):
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data

        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def bulid_vocab(sentences):
    word_counts = Counter(itertools.chain(*sentences))
    vocabulary_inv = [word[0] for word in word_counts.most_common()]  # 按词频构造字典
    vocabulary = {word: index for index, word in enumerate(vocabulary_inv)}
    return vocabulary, vocabulary_inv


def load_data(filename,max_length, cnum=1000):
    df = pd.read_csv(filename)
    df = df[:cnum]
    selected = ['Category', 'Text']
    non_selected = list(set(df.columns) - set(selected))

    df = df.drop(non_selected, axis=1)  # 去掉不需要的列
    df = df.dropna(axis=0, how='any', subset=selected)  # 去掉空行
    df = df.reindex(np.random.permutation(df.index))  # 打乱行顺序

    labels = sorted(list(set(df[selected[0]].tolist())))  # 分类标签
    num_labels = len(labels)
    one_hot = np.zeros((num_labels, num_labels), int)
    np.fill_diagonal(one_hot, 1)
    label_dict = dict(zip(labels, one_hot))

    x_raw = df[selected[1]].apply(lambda x: clean_str(x)).tolist()
    y_raw = df[selected[0]].apply(lambda y: label_dict[y]).tolist()

    x_raw = pad_sentences(x_raw,forced_sequence_length = max_length)
    vocabulary, vocabulary_inv = bulid_vocab(x_raw)

    x = np.array([[vocabulary[word] for word in sentence] for sentence in x_raw])
    y = np.array(y_raw)

    return x, y, vocabulary, vocabulary_inv, df, labels


def get_data(cnum=1000,test_size=0.9):
    training_config = '../training_config.json'
    params = json.loads(open(training_config, encoding='utf-8').read())

    input_file = params['data_path'] +'train.csv'
    x_, y_, vocabulary, vocabulary_inv, df, labels = load_data(input_file,params['sentence_size'] ,cnum=cnum)

    # 给每个单词分配一个256维度的向量
    word_embeddings = load_embeddings(vocabulary, params['word2vec_path'])
    # 构造输入矩阵
    embedding_mat = [word_embeddings[word] for index, word in enumerate(vocabulary_inv)]
    embedding_mat = np.array(embedding_mat, dtype=np.float32)

    # 将原始数据分割为训练数据和测试数据
    x, x_test, y, y_test = train_test_split(x_, y_, test_size=test_size)

    # 将训练数据分割为训练数据和验证数据
    x_train, x_val, y_train, y_val = train_test_split(x, y, test_size=test_size)

    logging.info('x_train:{},x_val:{},x_test:{}'.format(len(x_train), len(x_val), len(x_test)))
    logging.info('y_train:{},y_val:{},y_test:{}'.format(len(y_train), len(y_val), len(y_test)))

    return x_train,y_train,x_val,y_val,embedding_mat

def get_testdata(cnum=1000):
    training_config = '../training_config.json'
    params = json.loads(open(training_config, encoding='utf-8').read())

    input_file = params['data_path'] +'predict.csv'
    x, y, vocabulary, vocabulary_inv, df, labels = load_data(input_file,params['sentence_size'] ,cnum=cnum)
    return x, y