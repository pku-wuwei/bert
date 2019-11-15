#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: Wei Wu
@license: Apache Licence
@file: get_test_result.py
@time: 2019/11/14
@contact: wu.wei@pku.edu.cn
"""
from sklearn.metrics import f1_score
import numpy as np


def compare(predict_path, gold_path):
    preditions = []
    with open(predict_path) as fp:
        for line in fp:
            _, example_id, label_id, pos, neg = line.split()
            if label_id == '0':
                preditions.append([])
            preditions[-1].append(float(pos) > 0.5)
    golds = []
    with open(gold_path) as fg:
        for i, line in enumerate(fg):
            labels, text = line.strip().split('\t')
            gold_labels = [l == '1' for l in labels]
            golds.append(gold_labels)
    golds = np.array(golds)
    preditions = np.array(preditions)
    print(golds.shape)
    print(preditions.shape)
    print(f1_score(golds, preditions, average='micro'))


if __name__ == '__main__':
    gold_path = '/home/wuwei/glue_data/Reuters/test.tsv'
    pred_path = '/home/wuwei/bert/test_results.tsv'
    compare(pred_path, gold_path)
