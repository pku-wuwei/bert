#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: Wei Wu
@license: Apache Licence
@file: get_test_result.py
@time: 2019/11/14
@contact: wu.wei@pku.edu.cn
"""
from sklearn.metrics import f1_score, accuracy_score
import numpy as np
import tensorflow as tf
from sklearn.datasets import fetch_20newsgroups


def compare_multi(predict_path, gold_path):
    preditions = []
    with tf.gfile.GFile(predict_path) as fp:
        for line in fp:
            _, example_id, label_id, pos, neg = line.split()
            if label_id == '0':
                preditions.append([])
            preditions[-1].append(float(pos) > 0.5)
    golds = []
    with tf.gfile.GFile(gold_path) as fg:
        for i, line in enumerate(fg):
            labels, text = line.strip().split('\t')
            gold_labels = [l == '1' for l in labels]
            golds.append(gold_labels)
    golds = np.array(golds)
    preditions = np.array(preditions)
    print(golds.shape)
    print(preditions.shape)
    print(f1_score(golds, preditions, average='micro'))


def compare_multi1(predict_path, gold_path):
    preditions = []
    with tf.gfile.GFile(predict_path) as fp:
        for line in fp:
            data = [float(d) > 0.5 for d in line.strip().split()[3:]]
            preditions.append(data)
    preditions = np.array(preditions)
    golds = []
    with tf.gfile.GFile(gold_path) as fg:
        for i, line in enumerate(fg):
            labels, text = line.strip().split('\t')
            gold_labels = [l == '1' for l in labels]
            golds.append(gold_labels)
    golds = np.array(golds)
    print(golds.shape)
    print(preditions.shape)
    print(f1_score(golds, preditions, average='micro'))


def compare_single(predict_path):
    preditions = []
    with tf.gfile.GFile(predict_path) as fp:
        for line in fp:
            _, example_id, label_id, pos, neg = line.split()
            if label_id == '0':
                preditions.append([])
            preditions[-1].append(float(pos))
    predict_index = np.array([pred.index(max(pred)) for pred in preditions])
    # golds = fetch_20newsgroups(subset='test')['target']
    golds = []
    with open('/data/nfsdata/nlp/datasets/glue_data/SST-2/stsa.binary.test.txt') as fi:
        for line in fi:
            golds.append(int(line[0]))
    golds = np.array(golds)
    print(list(golds))
    print(list(predict_index))
    print(accuracy_score(golds, predict_index))


def compare_raw(predict_path):
    preditions = []
    with tf.gfile.GFile(predict_path) as fp:
        for line in fp:
            data = [float(d) for d in line.strip().split()[3:]]
            preditions.append(data.index(max(data)))
    preditions = np.array(preditions)
    golds = fetch_20newsgroups(subset='test')['target']
    print(golds)
    print(preditions)
    print(accuracy_score(golds, preditions))

def main1():
    gold_path = 'gs://shannon_albert/hedwig/Reuters/test.tsv'
    pred_path = 'gs://shannon_albert/reuters-raw-output/test_results.tsv'
    compare_multi1(pred_path, gold_path)


def main2():
    compare_single('gs://shannon_albert/20news-output/test_results.tsv')
    compare_raw('gs://shannon_albert/20news-raw-output/test_results.tsv')


if __name__ == '__main__':
    compare_single('sstlabel2-output/test_results.tsv')
