import numpy as np
from sklearn import datasets
import matplotlib.pyplot as plt
from scipy.stats import multivariate_normal
import csv


def Gaussian_Distribution(lst, M=1000, N=2, sigma=1):
    '''
    Parameters
    ----------
    N 维度
    M 样本数
    m 样本均值
    sigma: 样本方差

    Returns
    -------
    data  shape(M, N), M 个 N 维服从高斯分布的样本
    Gaussian  高斯分布概率密度函数
    '''
    mean = lst  # 均值矩阵，每个维度的均值都为 m
    cov = np.eye(N) * sigma  # 协方差矩阵，每个维度的方差都为 sigma

    # 产生 N 维高斯分布数据
    data = np.random.multivariate_normal(mean, cov, M)
    # N 维数据高斯分布概率密度函数
    Gaussian = multivariate_normal(mean=mean, cov=cov)

    return data, Gaussian


mean_1 = [1, 1, 1]
mean_2 = [-1, 1, 0]
mean_3 = [0, -1, 1]
num = 333
'''二元高斯散点图'''
data1, _ = Gaussian_Distribution(mean_1, M=num, N=3, sigma=1)
data2, _ = Gaussian_Distribution(mean_2, M=num+1, N=3, sigma=1)
data3, _ = Gaussian_Distribution(mean_3, M=num, N=3, sigma=2)
# x1, y1, z1 = data1.T
# x2, y2, z2 = data2.T
# x3, y3, z3 = data3.T
# print(x1, y1)
# plt.scatter(x1, y1)
# plt.scatter(x2, y2)
# plt.show()
path = "test_set_ex2_5.csv"


def create_csv():
    with open(path, 'w', newline="") as f:
        csv_write = csv.writer(f)
        csv_head = ["x", "y", "z", "L"]
        csv_write.writerow(csv_head)


create_csv()


for data in data1:
    with open(path, 'a+', newline="") as f:
        csv_write = csv.writer(f)
        data_row = [data[0], data[1], data[2], 1]
        csv_write.writerow(data_row)


for data in data2:
    with open(path, 'a+', newline="") as f:
        csv_write = csv.writer(f)
        data_row = [data[0], data[1], data[2], 2]
        csv_write.writerow(data_row)


for data in data3:
    with open(path, 'a+', newline="") as f:
        csv_write = csv.writer(f)
        data_row = [data[0], data[1], data[2], 3]
        csv_write.writerow(data_row)