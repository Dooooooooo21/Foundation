#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/10/20 16:48
# @Author  : dly
# @File    : baseline.py
# @Desc    :

import pandas as pd
import numpy as np
from category_encoders.target_encoder import TargetEncoder
from sklearn.model_selection import KFold
from sklearn.metrics import auc, roc_curve
from lightgbm import LGBMRegressor

data_path = 'C:/Users/Dooooooooo21/Desktop/project/FOUNDATION/'

# 导入数据
train = pd.read_csv(data_path + 'train.csv', index_col='id')
test = pd.read_csv(data_path + 'testA.csv', index_col='id')
target = train.pop('isDefault')
test = test[train.columns]

# 非数值列
s = train.apply(lambda x: x.dtype)
tecols = s[s == 'object'].index.tolist()


# 模型
def makelgb():
    lgbr = LGBMRegressor(num_leaves=30
                         , max_depth=5
                         , learning_rate=.02
                         , n_estimators=1000
                         , subsample_for_bin=5000
                         , min_child_samples=200
                         , colsample_bytree=.2
                         , reg_alpha=.1
                         , reg_lambda=.1
                         )
    return lgbr


# 本地验证
kf = KFold(n_splits=10, shuffle=True, random_state=100)
devscore = []
for tidx, didx in kf.split(train.index):
    tf = train.iloc[tidx]
    df = train.iloc[didx]
    tt = target.iloc[tidx]
    dt = target.iloc[didx]
    te = TargetEncoder(cols=tecols)
    tf = te.fit_transform(tf, tt)
    df = te.transform(df)
    lgbr = makelgb()
    lgbr.fit(tf, tt)
    pre = lgbr.predict(df)
    fpr, tpr, thresholds = roc_curve(dt, pre)
    score = auc(fpr, tpr)
    devscore.append(score)
print(np.mean(devscore))

# 在整个train集上重新训练，预测test，输出结果
lgbr = makelgb()
te = TargetEncoder(cols=tecols)
tf = te.fit_transform(train, target)
df = te.transform(test)
lgbr.fit(tf, target)
pre = lgbr.predict(df)
pd.Series(pre, name='isDefault', index=test.index).reset_index().to_csv('submit.csv', index=False)
