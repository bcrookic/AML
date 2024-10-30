import pickle

import pandas as pd
import torch
import torch.nn as nn

import numpy as np
import time
import sys

import os

filename = "../data/gcn_weights.pth"

if os.path.exists(filename):
    os.remove(filename)
    print("File deleted successfully")
else:
    print("File does not exist")


class GraphConv(nn.Module):
    def __init__(self, in_features, out_features, activation='relu', skip=False, skip_in_features=None):
        super(GraphConv, self).__init__()
        self.W = torch.nn.Parameter(torch.DoubleTensor(in_features, out_features))
        nn.init.xavier_uniform_(self.W)

        self.set_act = False
        if activation == 'relu':
            self.activation = nn.ReLU()
            self.set_act = True
        elif activation == 'softmax':
            self.activation = nn.Softmax(dim=1)
            self.set_act = True
        else:
            self.set_act = False
            raise ValueError("activations supported are 'relu' and 'softmax'")

        self.skip = skip
        if self.skip:
            if skip_in_features == None:
                raise ValueError("pass input feature size of the skip connection")
            self.W_skip = torch.nn.Parameter(torch.DoubleTensor(skip_in_features, out_features))
            nn.init.xavier_uniform_(self.W)

    def forward(self, A, H_in, H_skip_in=None):

        self.A = A
        self.H_in = H_in
        A_ = torch.add(self.A, torch.eye(self.A.shape[0]).double())
        D_ = torch.diag(A_.sum(1))

        D_root_inv = torch.inverse(torch.sqrt(D_))
        A_norm = torch.mm(torch.mm(D_root_inv, A_), D_root_inv)

        H_out = torch.mm(torch.mm(A_norm, H_in), self.W)

        if self.skip:
            H_skip_out = torch.mm(H_skip_in, self.W_skip)
            H_out = torch.add(H_out, H_skip_out)

        if self.set_act:
            H_out = self.activation(H_out)

        return H_out


class GCN_2layer(nn.Module):
    def __init__(self, in_features, hidden_features, out_features, skip=False):
        super(GCN_2layer, self).__init__()
        self.skip = skip

        self.gcl1 = GraphConv(in_features, hidden_features)

        if self.skip:
            self.gcl_skip = GraphConv(hidden_features, out_features, activation='softmax', skip=self.skip,
                                      skip_in_features=in_features)
        else:
            self.gcl2 = GraphConv(hidden_features, out_features, activation='softmax')

    def forward(self, A, X):
        out = self.gcl1(A, X)
        if self.skip:
            out = self.gcl_skip(A, out, X)
        else:
            out = self.gcl2(A, out)

        return out


path = './raw_data_pkl(+64re)/adj_mats.pkl'
f = open(path, 'rb')
ass = pickle.load(f)

path = './raw_data_pkl(+64re)/features_labelled_ts.pkl'
f = open(path, 'rb')
fss = pickle.load(f)

path = './raw_data_pkl(+64re)/classes_ts.pkl'
f = open(path, 'rb')
css = pickle.load(f)

num_features = fss[0].shape[1]
num_hiddens = num_features // 2
num_classes = 2
num_ts = 49
epochs = 15
lr = 0.001
max_train_ts = 34
train_ts = np.arange(max_train_ts)

adj_mats, features_labelled_ts, classes_ts = ass[0:35], fss[0:35], css[0:35]

labels_ts = []
for c in classes_ts:
    labels_ts.append(np.array(c['class'] == '2', dtype=np.long))

gcn = GCN_2layer(num_features, num_hiddens, num_classes)
train_loss = nn.CrossEntropyLoss(weight=torch.DoubleTensor([0.7, 0.3]))
optimizer = torch.optim.Adam(gcn.parameters(), lr=lr)

for ts in train_ts:
    A = torch.tensor(adj_mats[ts].values)
    X = torch.tensor(features_labelled_ts[ts].values)
    L = torch.tensor(labels_ts[ts], dtype=torch.long)
    for ep in range(epochs):
        t_start = time.time()

        gcn.train()
        optimizer.zero_grad()
        out = gcn(A, X)

        loss = train_loss(out, L)
        train_pred = out.max(1)[1].type_as(L)
        acc = (train_pred.eq(L).double().sum()) / L.shape[0]

        loss.backward()
        optimizer.step()

        sys.stdout.write("\r Epoch %d/%d Timestamp %d/%d training loss: %f training accuracy: %f Time: %s"
                         % (ep, epochs, ts, max_train_ts, loss, acc, time.time() - t_start)
                         )

torch.save(gcn.state_dict(), str(filename))

test_ts = np.arange(14)
adj_mats, features_labelled_ts, classes_ts = ass[35:49], fss[35:49], css[35:49]

# id = []
labels_ts = []
for c in classes_ts:
    labels_ts.append(np.array(c['class'] == '2', dtype=np.long))
    # id.extend(c.index.tolist())

gcn = GCN_2layer(num_features, num_hiddens, num_classes)
gcn.load_state_dict(torch.load(filename))

from sklearn.metrics import precision_score, recall_score, f1_score

# Testing
test_accs = []
test_precisions = []
test_recalls = []
test_f1s = []

y_true = []
y_pred = []

for ts in test_ts:
    A = torch.tensor(adj_mats[ts].values)
    X = torch.tensor(features_labelled_ts[ts].values)
    L = torch.tensor(labels_ts[ts], dtype=torch.long)

    gcn.eval()
    test_out = gcn(A, X)

    test_pred = test_out.max(1)[1].type_as(L)
    t_acc = (test_pred.eq(L).double().sum()) / L.shape[0]
    test_accs.append(t_acc.item())

    y_true.extend(L.tolist())
    y_pred.extend(test_pred.tolist())

# macro
# micro_acc = np.array(test_accs).mean()
macro_prec = precision_score(y_true, y_pred, average='macro')
macro_rec = recall_score(y_true, y_pred, average='macro')
macro_f1 = f1_score(y_true, y_pred, average='macro')

print(macro_prec)
print(macro_rec)
print(macro_f1)




# import csv
#
# filename = "4.csv"  # CSV文件名
# target_values = ["FP", "FN"]  # 目标数值列表
# id_list = []  # 存储第一列对应的数据
#
# with open(filename, 'r') as file:
#     reader = csv.reader(file)
#
#     for row in reader:
#         if len(row) >= 3 and row[2] in target_values:
#             id_list.append(int(row[0]))
#
#
# filename = "edgelist.csv"  # CSV文件名
#
# filtered_rows = []  # 存储满足条件的行
#
# with open(filename, 'r') as file:
#     reader = csv.reader(file)
#
#     for row in reader:
#         if len(row) >= 2 and (int(row[0]) in id_list or int(row[1]) in id_list):
#             filtered_rows.append(row)
#
# # 将满足条件的行重新写入CSV文件
# with open('2.csv', 'w', newline='') as file:
#     writer = csv.writer(file)
#     writer.writerows(filtered_rows)




# import csv


# results = []
# for i, pred in enumerate(y_pred):
#     true_label = y_true[i]
#     if pred == true_label:
#         if pred == 1:
#             result = 'TN'
#         else:
#             result = 'TF'
#     else:
#         if pred == 1:
#             result = 'FP'
#         else:
#             result = 'FN'
#     results.append((id[i], pred, result))
#
# with open('4.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['ID', 'Prediction', 'Category'])
#     writer.writerows(results)






# # 使用 concat() 函数将多个 DataFrame 按行合并
# merged_f = pd.concat(features_labelled_ts, axis=0)
#
# # 将合并后的 DataFrame 写入 CSV 文件
# merged_f_df.to_csv('1.csv', header=False)





# # 使用 concat() 函数将多个 DataFrame 按行合并
# merged_c = pd.concat(classes_ts, axis=0)
#
# # 对合并后的 DataFrame 的第二列进行减一操作
# merged_c.iloc[:, 1] = merged_c.iloc[:, 1].apply(lambda x: x - 1)
#
# # 将合并后的 DataFrame 写入 CSV 文件
# merged_c.to_csv('3.csv', header=False)



# results = []
# for i, pred in enumerate(y_pred):
#     true_label = y_true[i]
#     if pred == true_label:
#         if pred == 1:
#             result = 'TN'
#         else:
#             result = 'TF'
#     else:
#         if pred == 1:
#             result = 'FP'
#         else:
#             result = 'FN'
#     results.append((id[i], pred, result))
#
# with open('5.csv', 'w', newline='') as csvfile:
#     writer = csv.writer(csvfile)
#     writer.writerow(['ID', 'Prediction', 'Category'])
#     writer.writerows(results)



