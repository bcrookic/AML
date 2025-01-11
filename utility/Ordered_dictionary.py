import pickle

with open('./dataset/embedding_sim.pkl', 'rb') as file:
    data = pickle.load(file)

sorted_keys = sorted(data.keys(), key=lambda x: int(x))

from collections import OrderedDict

ordered_dict = OrderedDict()

for key in sorted_keys:
    ordered_dict[key] = data[key]

with open('./data/reduced_embedding_sim.pkl', 'wb') as file:
    pickle.dump(ordered_dict, file)
