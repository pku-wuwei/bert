#!/usr/bin/env python
# -*- coding:utf-8 _*-
"""
@author: Wei Wu
@license: Apache Licence
@file: label_construction.py
@time: 2019/11/14
@contact: wu.wei@pku.edu.cn

将分类标签构造成问题
"""
import json

import wikipedia
from nltk.corpus import reuters
from sklearn.datasets import fetch_20newsgroups


class LabelConstruction(object):

    def __init__(self, mode, dataset):
        self.mode = mode
        self.dataset = dataset

    def construct_label_query_map(self):
        if self.mode == 'desc1':
            if self.dataset == 'sst':
                return {'0': 'negative', '1': 'positive'}
            elif self.dataset == 'reuters':
                return reuters.categories()
            elif self.dataset == '20news':
                word_map = {'rec': 'recreation',
                            'sci': 'science',
                            'comp': 'company',
                            'crypt': 'encryption',
                            'med': 'medicine'
                            }
                labels = fetch_20newsgroups(subset='test')['target_names']
                label_map = [[word_map[w] if w in word_map else w for w in l.split('.')] for l in labels]
                return label_map
        elif self.mode == 'wiki1':
            if self.dataset == 'reuters':
                with open('cat_wiki.json') as fi:
                    return json.load(fi)

    def get_reuters_wiki_desc(self):
        cat2wiki = {}
        parent = ''
        with open('cat_desc.txt') as fi:
            for line in fi:
                if line.startswith('**'):
                    parent = line[: line.index('(')].strip('*').strip()
                elif line.strip():
                    if line.find('(') == -1:
                        cat = line.strip()
                        cat2wiki[cat.lower()] = [cat, parent]
                    else:
                        cat = line[line.index('(') + 1: line.index(')')]
                        desc = line.replace('(' + cat + ')', '').strip()
                        cat2wiki[cat.lower()] = [desc, parent]
        cat_map = {
            'earn': 'earnings',
            'sun-meal': 'meal',
            'orange': 'Orange (fruit)',
            'cocoa': 'Theobroma cacao',
            'jobs': 'job',
            'instal-debt': 'Installment Debt',
            'groundnut': 'peanut',
            'jet': 'Jet aircraft',
            'money-fx': 'Foreign Exchange',
            'reserves': 'Reserve (accounting)',
            'carcass': 'Carrion',
            'palmkernel': 'palm kernel',
            'hog': 'domestic pig',
            'meal-feed': 'Animal feed',
            'dmk': 'Deutsche Mark',
            'veg-oil': 'Vegetable oil',
            'sunseed': 'sunflower seed'
        }
        with open('cat_wiki.json', 'w') as fo:
            wiki_desc = []
            for cat in reuters.categories():
                print(cat)
                wiki = wikipedia.summary(cat_map[cat]) if cat in cat_map else wikipedia.summary(cat2wiki[cat][0])
                wiki_desc.append([cat, cat2wiki[cat][0], cat2wiki[cat][1], wiki])
            json.dump(wiki_desc, fo, indent=2, ensure_ascii=False)

    def get_20_news_dataset(self):
        data = []
        newsgroups_train = fetch_20newsgroups(subset='test')
        for text, label in zip(newsgroups_train['data'], newsgroups_train['target']):
            data.append([text, newsgroups_train['target_names'][label]])
        with open('20news/dev.json', 'w') as fo:
            json.dump(data, fo, ensure_ascii=False, indent=2)


if __name__ == '__main__':
    constructer = LabelConstruction('desc1', '20news')
    m = constructer.construct_label_query_map()
    print(m)
