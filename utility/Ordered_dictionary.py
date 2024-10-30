import pickle

# 加载.pkl文件并反序列化为字典对象
with open('embedding_sim.pkl', 'rb') as file:
    data = pickle.load(file)

# 对字典的键进行排序并转换为整数
sorted_keys = sorted(data.keys(), key=lambda x: int(x))

from collections import OrderedDict

# 创建一个新的有序字典
ordered_dict = OrderedDict()

# 逐个添加排序后的键值对到新字典
for key in sorted_keys:
    ordered_dict[key] = data[key]

with open('reduced_embedding_sim.pkl', 'wb') as file:
    pickle.dump(ordered_dict, file)