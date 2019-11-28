#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: Wei Wu
@license: Apache Licence
@file: eval_ner.py
@time: 2019/11/28
@contact: wu.wei@pku.edu.cn
"""
from seqeval.metrics import accuracy_score, classification_report, f1_score


def transform_tag(prev_tag: str) -> str:
    all_tags = ['O', 'B-NR', 'B-NS', 'B-NT', 'E-NR', 'E-NS', 'E-NT', 'M-NR', 'M-NS', 'M-NT', 'S-NR', 'S-NS', 'S-NT']
    if isinstance(prev_tag, int):
        prev_tag = all_tags[prev_tag]
    if prev_tag.startswith('M-') or prev_tag.startswith('E-'):
        return prev_tag.replace('M-', 'I-').replace('E-', 'I-')
    elif prev_tag.startswith('S-'):
        return prev_tag.replace('S-', 'B-')
    else:
        return prev_tag


def get_evaluation(gold_file, pred_file):

    golds = []
    with open(gold_file) as fi:
        for line in fi:
            data, tags = line.strip().split('\t')
            golds.append([transform_tag(tag) for tag in tags.split()])
    preds = []
    with open(pred_file) as fi:
        for i, line in enumerate(fi):
            data = line.strip().split()
            pred = data[2: 2 + len(golds[i])]
            preds.append([transform_tag(int(tag)) for tag in pred])
    for i, (p, g) in enumerate(zip(preds, golds)):
        if p != g:
            print(F"{i}\n{g}\n{p}\n\n")
    print(classification_report(golds, preds))
    print(f1_score(golds, preds))


def main():
    gold_file = '/data/nfsdata/nlp/datasets/sequence_labeling/msra/test.tsv'
    pred_file = '/data/nfs/wuwei/study/bert/msra-bert-crf/test_results.tsv'
    get_evaluation(gold_file, pred_file)


if __name__ == '__main__':
    main()
