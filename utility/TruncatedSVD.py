import pickle
from sklearn.decomposition import TruncatedSVD

with open('../data/reduced_embedding_sim.pkl', 'rb') as file:
    data = pickle.load(file)

matrix = list(data.values())

n = 64  # dimensions

svd = TruncatedSVD(n_components=n)

matrix_n_d = svd.fit_transform(matrix)

reduced_dict = {str(i): matrix_n_d[i] for i in range(len(matrix_n_d))}

with open('../data/reduced_dict.pkl', 'wb') as file:
    pickle.dump(reduced_dict, file)

