# -*- coding: utf-8 -*-

import pandas as pd
import numpy as np
from sklearn.cluster import KMeans

train_file_path = 'train.csv'
test_file_path = 'test.csv'


train_df = pd.read_csv(filepath_or_buffer=train_file_path)

test_df = pd.read_csv(filepath_or_buffer=test_file_path)

# 针对测试样本添加target列，与训练样本对齐，全部置为0
col_name = test_df.columns.tolist()
col_name.insert(col_name.index('ID_code') + 1, 'target')
test_df = test_df.reindex(columns=col_name)
test_df['target'] = 0

# 合并训练与测试样本
total_df = pd.concat([train_df, test_df], axis=0)

# 针对特征列进行正态归一化操作
df1 = total_df[total_df.columns.tolist()[0:2]]
df2 = total_df[total_df.columns.tolist()[2:]]

# 按列进行正态归一化
print('the columns normalization')
df2 = df2.apply(func=lambda x: (x - np.mean(x)) / (np.std(x)))

# 按行进行归一化,纯粹是将欧式距离与cos相似距离统一，不加试试
print('the row normalization')
df2 = df2.apply(func=lambda x: x*1.0/np.sqrt(np.sum(x*x)), axis=1)
df = pd.concat([df1, df2], axis=1)

print(df)

# 聚类，类别拍了个200，具体参数可参考KMeans
# n_init表示选择多少组初始聚类点，从n组中根据误差选择最优的一种
# n_jobs表示多少个并行化，根据机器内存来定
print('the cluster id beginning')
cluster_data = df2.values
kmeans = KMeans(n_clusters=200, verbose=1, n_init=1, n_jobs=1)
kmeans.fit(cluster_data)

#  打印结果
print(kmeans.labels_)
print(kmeans.cluster_centers_)
print(kmeans.inertia_)
print(np.shape(kmeans.labels_))
print(np.shape(kmeans.cluster_centers_))

# 将样本和聚类结果写入文件
df['cluster'] = np.array(kmeans.labels_)
df.to_csv(path_or_buf='total_sample_result.csv', index=False, header=True)

# 将聚类向量写入文件
df_cluster = pd.DataFrame(np.array(kmeans.cluster_centers_))
df_cluster.to_csv(path_or_buf='cluster.csv', index=True, header=False)




