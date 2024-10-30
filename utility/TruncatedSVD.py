# -*- coding = utf-8 -*-
# @Time : 2023/8/12 15:59
# @File : TruncatedSVD.py
# @Software : PyCharm.py

import pickle
from sklearn.decomposition import TruncatedSVD

with open('pkl_data/ordered_embedding_sim.pkl', 'rb') as file:
    data = pickle.load(file)

matrix = list(data.values())

n = 64

svd = TruncatedSVD(n_components=n)

matrix_n_d = svd.fit_transform(matrix)

reduced_dict = {str(i): matrix_n_d[i] for i in range(len(matrix_n_d))}

with open('../data/reduced_dict.pkl', 'wb') as file:
    pickle.dump(reduced_dict, file)
