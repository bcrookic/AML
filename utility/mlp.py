import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # 激励函数都在这

path = './raw_data_pkl(+16re)/features_labelled_ts.pkl'  # pkl文件所在路径
f = open(path, 'rb')
fss = pickle.load(f)

path = './raw_data_pkl(+16re)/classes_ts.pkl'  # pkl文件所在路径
f = open(path, 'rb')
css = pickle.load(f)

# # 保留每个DataFrame的前94列（LF）
# fss = [df.iloc[:, :94] for df in fss]

# 保留每个DataFrame的前94列和后16列，并重新设置列名（LF）
# fss = [pd.concat([df.iloc[:, :94], df.iloc[:, -16:].rename(columns=lambda x: x - 72)], axis=1) for df in fss]




def get_xy(i):
    features, classes = fss[i], css[i]

    # 将x_DataFrame 列中的字符串转换为张量
    x_tensor_list = [torch.tensor(features[col].astype(float).values, dtype=torch.float32) for col in features.columns]
    features = torch.stack(x_tensor_list, dim=1)  # 形状为 (x, y)，其中 x 是行数，y 是列数

    # 将y_DataFrame 中的字符串转换为 long 类型的张量，并进行值映射
    classes = torch.tensor(classes['class'].astype(int).values - 1, dtype=torch.long)  # 形状为 (x,)

    return features, classes


# 建立神经网络
class Net(torch.nn.Module):  # 继承 torch 的 Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  # 继承 __init__ 功能
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  # 隐藏层线性输出
        self.out = torch.nn.Linear(n_hidden, n_output)  # 输出层线性输出

    def forward(self, x):
        # 正向传播输入值, 神经网络分析出输出值
        x = F.relu(self.hidden(x))  # 激励函数(隐藏层的线性值)
        x = self.out(x)  # 输出值, 但是这个不是预测值, 预测值还需要再另外计算
        return x


f = fss[0].shape[1]
h = f // 2
net = Net(n_feature=f, n_hidden=h, n_output=2)  # 几个类别就几个 output；n_hidden用n_feature的一半

# 训练网络

#
# import time
#
# # 记录开始时间
# start_time = time.time()

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
# 算误差的时候, 注意真实值!不是! one-hot 形式的, 而是1D Tensor, (batch,)
# 但是预测值是2D tensor (batch, n_classes)
loss_func = torch.nn.CrossEntropyLoss()

for i in range(0, 35):
    x, y = get_xy(i)

    for t in range(100):
        out = net(x)  # 喂给 net 训练数据 x, 输出分析值

        loss = loss_func(out, y)  # 计算两者的误差

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

# # 记录结束时间
# end_time = time.time()
#
# # 计算训练时间
# training_time = end_time - start_time

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# # testing
#
# accuracies = []
# precisions = []
# recalls = []
# f1_scores = []
# # tp_list = []
# # tn_list = []
# # fp_list = []
# # fn_list = []
#
# for i in range(35, 49):
#     x, y = get_xy(i)
#
#     with torch.no_grad():
#         net.eval()  # 将模型设置为评估模式
#         # 进行单次测试，并获得准确率、精确率、召回率和F1值
#         out = net(x)
#         prediction = torch.max(F.softmax(out), 1)[1]
#         pred_y = prediction.data.numpy().squeeze()
#         target_y = y.data.numpy()
#
#     accuracy = accuracy_score(target_y, pred_y)
#     precision = precision_score(target_y, pred_y)
#     recall = recall_score(target_y, pred_y)
#     f1 = f1_score(target_y, pred_y)
#
#     # 计算TP、TN、FP和FN
#     # tp = np.sum(np.logical_and(pred_y == 1, target_y == 1))
#     # tn = np.sum(np.logical_and(pred_y == 0, target_y == 0))
#     # fp = np.sum(np.logical_and(pred_y == 1, target_y == 0))
#     # fn = np.sum(np.logical_and(pred_y == 0, target_y == 1))
#
#     # 将指标值添加到列表中
#     accuracies.append(accuracy)
#     precisions.append(precision)
#     recalls.append(recall)
#     f1_scores.append(f1)
#     # tp_list.append(tp)
#     # tn_list.append(tn)
#     # fp_list.append(fp)
#     # fn_list.append(fn)
#
# # 计算平均值
# accuracy = np.mean(accuracies)
# precision = np.mean(precisions)
# recall = np.mean(recalls)
# f1 = np.mean(f1_scores)
# # tp = np.sum(tp_list)
# # tn = np.sum(tn_list)
# # fp = np.sum(fp_list)
# # fn = np.sum(fn_list)
#
# print("Averaged Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1:", f1)
# # print("TP:", tp)
# # print("TN:", tn)
# # print("FP:", fp)
# # print("FN:", fn)


# # testing
#
# accuracies = []
# precisions = []
# recalls = []
# f1_scores = []
#
# for i in range(35, 49):
#     x, y = get_xy(i)
#
#     with torch.no_grad():
#         net.eval()  # 将模型设置为评估模式
#         # 进行单次测试，并获得准确率、精确率、召回率和F1值
#         out = net(x)
#         prediction = torch.max(F.softmax(out), 1)[1]
#         pred_y = prediction.data.numpy().squeeze()
#         target_y = y.data.numpy()
#
#     accuracy = accuracy_score(target_y, pred_y)
#     precision = precision_score(target_y, pred_y)
#     recall = recall_score(target_y, pred_y)
#     f1 = f1_score(target_y, pred_y)
#
#     # 将指标值添加到列表中
#     accuracies.append(accuracy)
#     precisions.append(precision)
#     recalls.append(recall)
#     f1_scores.append(f1)
#
# # 计算平均值
# accuracy = np.mean(accuracies)
# precision = np.mean(precisions)
# recall = np.mean(recalls)
# f1 = np.mean(f1_scores)
#
# print("Averaged Accuracy:", accuracy)
# print("Precision:", precision)
# print("Recall:", recall)
# print("F1:", f1)

from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
import numpy as np

accuracies = []
precisions_micro = []
recalls_micro = []
f1_scores_micro = []
precisions_macro = []
recalls_macro = []
f1_scores_macro = []

for i in range(35, 49):
    x, y = get_xy(i)

    with torch.no_grad():
        net.eval()
        out = net(x)
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()

    accuracy = accuracy_score(target_y, pred_y)
    precision_micro = precision_score(target_y, pred_y, average='micro')
    recall_micro = recall_score(target_y, pred_y, average='micro')
    f1_micro = f1_score(target_y, pred_y, average='micro')
    precision_macro = precision_score(target_y, pred_y, average='macro')
    recall_macro = recall_score(target_y, pred_y, average='macro')
    f1_macro = f1_score(target_y, pred_y, average='macro')

    accuracies.append(accuracy)
    precisions_micro.append(precision_micro)
    recalls_micro.append(recall_micro)
    f1_scores_micro.append(f1_micro)
    precisions_macro.append(precision_macro)
    recalls_macro.append(recall_macro)
    f1_scores_macro.append(f1_macro)

accuracy = np.mean(accuracies)
precision_micro = np.mean(precisions_micro)
recall_micro = np.mean(recalls_micro)
f1_micro = np.mean(f1_scores_micro)
precision_macro = np.mean(precisions_macro)
recall_macro = np.mean(recalls_macro)
f1_macro = np.mean(f1_scores_macro)

print("Averaged Accuracy:", accuracy)
print("Micro Precision:", precision_micro)
print("Micro Recall:", recall_micro)
print("Micro F1:", f1_micro)
print("---------------------------------")
print("Averaged Accuracy:", accuracy)
print("Macro Precision:", precision_macro)
print("Macro Recall:", recall_macro)
print("Macro F1:", f1_macro)
