import pandas as pd
import numpy as np
import torch
from sklearn.preprocessing import MinMaxScaler
from sklearn.linear_model import LogisticRegression
from sklearn.metrics import recall_score,precision_score
from sklearn.linear_model import Ridge
import copy
import torch
import torch.nn.functional as F
import copy
from torch import nn, optim
import torch.utils.data as Data
import time
import math
def read(path):
    df = pd.read_excel(path,index_col=0)
    df.columns = np.arange(len(df.loc['U_8126464',:]))
    df.index = np.arange(len(df))
    return df
#data是交互次数矩阵，data01是是否交互的矩阵
data = read('user_course_交互次数.xlsx')
data01 = read('user_course_matrix.xlsx')
#随机选取一些交互次数来当x，一些是否交互当y
np.random.seed(2023)
user_mask = np.arange(len(data))
np.random.shuffle(user_mask)
cour_mask = np.arange(len(data.loc[0,:]))
np.random.shuffle(cour_mask)
x_train,y_train = data.loc[user_mask[:1500],cour_mask[:250]],data01.loc[user_mask[:1500],cour_mask[250:]]
x_test,y_test = data.loc[user_mask[1500:],cour_mask[:250]],data01.loc[user_mask[1500:],cour_mask[250:]]
scale = MinMaxScaler()
x_train = scale.fit_transform(x_train)
x_test = scale.transform(x_test)
# y_train = np.array(y_train)
# y_test = np.array(y_test)
estimator = Ridge()
z = 0 #预测对的1的个数
y_predicts = []
y_tests = []
for i in cour_mask[250:]:
    if np.sum(y_train.loc[:,i])!=0:
        estimator.fit(x_train, np.array(y_train.loc[:,i]))
        y_predict = estimator.predict(x_test)
        y_predicts.append(list(y_predict))
        y_tests.append(list(y_test.loc[:,i]))
#行是人，列是课程
y_predicts = np.array(y_predicts).T
y_tests = np.array(y_tests).T
n = 15 #选n个课出来
row = 0
True_positive = 0 #选对的1的个数
for y_predict in y_predicts:
    y_predict = list(y_predict)
    y_predict_copy = copy.deepcopy(y_predict)
    y_predict.sort(reverse=True)
    for m in range(n):
        True_positive += y_tests[row,y_predict_copy.index(y_predict[m])]
    row += 1
print(True_positive/(n*822))
#概率归一
y_predicts = F.softmax(torch.Tensor(y_predicts))
y_predicts = pd.DataFrame(y_predicts)
y_tests = pd.DataFrame(y_tests)
writer1 = pd.ExcelWriter('y_predict.xlsx')
y_predicts.to_excel(writer1)
writer2 = pd.ExcelWriter('y_test.xlsx')
y_tests.to_excel(writer2)
writer1.close()
writer2.close()