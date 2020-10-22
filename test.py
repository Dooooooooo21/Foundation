#!/usr/bin/python3.7
# -*- coding: utf-8 -*-
# @Time    : 2020/10/21 13:48
# @Author  : dly
# @File    : test.py
# @Desc    :

import numpy as np
from sklearn import metrics
import matplotlib.pyplot as plt

y_pred = [0, 1, 1, 0, 1, 1, 0, 1, 1, 1]
y_true = [0, 1, 1, 0, 1, 0, 1, 1, 0, 1]

print('混淆矩阵\n', metrics.confusion_matrix(y_true, y_pred))

print('ACC', metrics.accuracy_score(y_true, y_pred))
print('P', metrics.precision_score(y_true, y_pred))
print('R', metrics.recall_score(y_true, y_pred))
print('F1', metrics.f1_score(y_true, y_pred))


# PR曲线
def pr():
    p, r, t = metrics.precision_recall_curve(y_true, y_pred)
    plt.plot(p, r)
    plt.show()


# ROC曲线
def roc():
    fpr, tpr, th = metrics.roc_curve(y_true, y_pred)
    plt.title('ROC')
    plt.plot(fpr, tpr, 'b')
    plt.xlabel('FPR')
    plt.ylabel('TPR')
    plt.show()
    print('AUC score', metrics.roc_auc_score(y_true, y_pred))


roc()
