#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/10/23 10:04
# @Author  : dly
# @File    : EDA.py
# @Desc    :

import numpy as np
import pandas as pd
from sklearn.model_selection import KFold
from sklearn.preprocessing import LabelEncoder
from sklearn.feature_selection import SelectKBest
from lightgbm import LGBMRegressor
import matplotlib.pyplot as plt
import seaborn as sns
import pandas_profiling
from tqdm import tqdm

# pd.options.display.max_columns = None

data_path = 'C:/Users/Dooooooooo21/Desktop/project/FOUNDATION/'

train_all = pd.read_csv(data_path + 'train.csv')
test = pd.read_csv(data_path + 'testA.csv')

print(f'There are {train_all.isnull().any().sum()} columns in train dataset with missing values.')


# 查看缺失率大于 50% 的特征
def fea_n_half():
    have_null_fea_dict = (train_all.isnull().sum() / len(train_all)).to_dict()
    fea_null_morethanhalf = {}

    for k, v in have_null_fea_dict.items():
        if v > 0.5:
            fea_null_morethanhalf[k] = v

    print(fea_null_morethanhalf)


# 查看缺失特征及缺失率
def visual_fea_nul():
    missing = train_all.isnull().sum() / len(train_all)
    missing = missing[missing > 0]
    missing.sort_values(inplace=True)
    missing.plot.bar()
    plt.show()


# 查看单一值的特征
one_value_fea = [col for col in train_all.columns if train_all[col].nunique() <= 1]
print(one_value_fea)

# 区分数值型特征和类别特征

numrical_fea = list(train_all.select_dtypes(exclude=['object']).columns)
category_fea = list(filter(lambda x: x not in numrical_fea, list(train_all.columns)))


# 数值型特征：离散型和连续型
# 过滤数值型特征
def get_numrical_serial_fea(data, feas):
    numrical_serial_feas = []
    numrical_noserial_feas = []

    for fea in feas:
        tmp = data[fea].nunique()
        if tmp <= 10:
            numrical_noserial_feas.append(fea)
            continue
        numrical_serial_feas.append(fea)

    return numrical_serial_feas, numrical_noserial_feas


numrical_serial_feas, numrical_noserial_feas = get_numrical_serial_fea(train_all, numrical_fea)


# todo 离散型变量，相差悬殊的，再分析用不用


# 连续型变量特征分布
def numrical_serial():
    f = pd.melt(train_all, value_vars='interestRate')
    g = sns.FacetGrid(f, col='variable', col_wrap=2, sharex=False, sharey=False)
    g = g.map(sns.distplot, 'value')
    plt.show()


# todo 查看变量是否符合正态分布，不符合的可以log后再观察是否符合正太分布
# todo 正态可以让模型更快收敛，提高训练速度


# 单一变量分布
def single_fea():
    plt.figure(figsize=(8, 8))
    sns.barplot(train_all['employmentLength'].value_counts(dropna=False),
                train_all["employmentLength"].value_counts(dropna=False).keys())
    plt.show()


# 数据报告
# pf = pandas_profiling.ProfileReport(train_all)
# pf.to_file('./report.html')


# 数据预处理：异常值、时间格式、对象类型转数值
# 异常值处理：均方差、箱型图
# 数据分箱
# 特征交互
# 特征编码
# 特征选择


def employmentLength_to_int(s):
    if pd.isnull(s):
        return s

    return np.int8(s.split()[0])


for data in [train_all, test]:
    data['employmentLength'].replace(to_replace='10+ years', value='10 years', inplace=True)
    data['employmentLength'].replace(to_replace='< 1 year', value='0 years', inplace=True)
    data['employmentLength'] = data['employmentLength'].apply(employmentLength_to_int)
    data['earliesCreditLine'] = data['earliesCreditLine'].apply(lambda s: int(s[-4:]))
    data['grade'] = data['grade'].map({'A': 1, 'B': 2, 'C': 3, 'D': 4, 'E': 5, 'F': 6, 'G': 7})
    data = pd.get_dummies(data, columns=['subGrade', 'homeOwnership', 'verificationStatus', 'purpose', 'regionCode'],
                          drop_first=True)

print(data['employmentLength'].value_counts(dropna=False).sort_index())
print(data['earliesCreditLine'].value_counts(dropna=False).sort_index())

print(data.head())


# 在统计学中，如果一个数据分布近似正态，那么大约 68% 的数据值会在均值的一个标准差范围内，
# 大约 95% 会在两个标准差范围内，大约 99.7% 会在三个标准差范围内


def find_outliers_by_3segama(data, fea):
    data_std = np.std(data[fea])
    data_mean = np.mean(data[fea])

    outlier_cut_off = data_std * 3

    lower = data_mean - outlier_cut_off
    higher = data_mean + outlier_cut_off

    data[fea + '_outliers'] = data[fea].apply(lambda x: str('异常值') if x > higher or x < lower else '正常值')

    return data


data_train = train_all.copy()
for fea in numrical_fea:
    data_train = find_outliers_by_3segama(data_train, fea)
    print(data_train[fea + '_outliers'].value_counts())
    print(data_train.groupby(fea + '_outliers')['isDefault'].sum())
    print('*' * 10)

for col in tqdm(['employmentTitle', 'postCode', 'title', 'subGrade']):
    le = LabelEncoder()
    le.fit(list(data_train[col].astype(str).values) + list(test[col].astype(str).values))
    data_train[col] = le.transform(list(data_train[col].astype(str).values))
    test[col] = le.transform(list(test[col].astype(str).values))

# todo 特征选择技术可以精简掉无用的特征，以降低最终模型的复杂性，它的最终目的是得到一个简约模型，
#  在不降低预测准确率或对预测准确率影响不大的情况下提高计算速度。特征选择不是为了减少训练时间
#  （实际上，一些技术会增加总体训练时间），而是为了减少模型评分时间

target = train_all.pop('isDefault')
# fea_select = SelectKBest(k=5).fit_transform(train_all[numrical_serial_feas], target)
# print(fea_select)
