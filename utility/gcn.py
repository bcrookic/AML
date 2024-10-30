import pickle

import pandas as pd
import torch
import torch.nn as nn

import numpy as np
import time
import sys

# Importing the os library
import os

from sklearn.metrics import f1_score, precision_score, recall_score

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
        # A must be an n x n matrix as it is an adjacency matrix
        # H is the input of the node embeddings, shape will n x in_features
        self.A = A
        self.H_in = H_in
        A_ = torch.add(self.A, torch.eye(self.A.shape[0]).double())
        D_ = torch.diag(A_.sum(1))
        # since D_ is a diagonal matrix,
        # its root will be the roots of the diagonal elements on the principle diagonal
        # since A is an adjacency matrix, we are only dealing with positive values
        # all roots will be real
        D_root_inv = torch.inverse(torch.sqrt(D_))
        A_norm = torch.mm(torch.mm(D_root_inv, A_), D_root_inv)
        # shape of A_norm will be n x n

        H_out = torch.mm(torch.mm(A_norm, H_in), self.W)
        # shape of H_out will be n x out_features

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


path = './raw_data_pkl/adj_mats.pkl'  # pkl文件所在路径
f = open(path, 'rb')
ass = pickle.load(f)

path = './raw_data_pkl/features_labelled_ts.pkl'  # pkl文件所在路径
f = open(path, 'rb')
fss = pickle.load(f)

path = './raw_data_pkl/classes_ts.pkl'  # pkl文件所在路径
f = open(path, 'rb')
css = pickle.load(f)

# # # 保留每个DataFrame的前94列
# fss = [df.iloc[:, :94] for df in fss]

# 保留每个DataFrame的前94列和后16列，并重新设置列名
# fss = [pd.concat([df.iloc[:, :94], df.iloc[:, -16:].rename(columns=lambda x: x - 72)], axis=1) for df in fss]

num_features = fss[0].shape[1]
num_hiddens = num_features // 2  # num_features // 2
num_classes = 2
num_ts = 49
epochs = 15
lr = 0.001
max_train_ts = 34
train_ts = np.arange(max_train_ts)

adj_mats, features_labelled_ts, classes_ts = ass[0:35], fss[0:35], css[0:35]

# 0 - illicit, 1 - licit
labels_ts = []
for c in classes_ts:
    labels_ts.append(np.array(c['class'] == '2', dtype=np.long))

gcn = GCN_2layer(num_features, num_hiddens, num_classes)
train_loss = nn.CrossEntropyLoss(weight=torch.DoubleTensor([0.7, 0.3]))
optimizer = torch.optim.Adam(gcn.parameters(), lr=lr)

# Training

# import time

# # 记录开始时间
# start_time = time.time()

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

# # 记录结束时间
# end_time = time.time()
#
# # 计算训练时间
# training_time = end_time - start_time

torch.save(gcn.state_dict(), str("gcn_weights.pth"))

# from sklearn.metrics import f1_score, precision_score, recall_score
# from sklearn.metrics import confusion_matrix

test_ts = np.arange(14)
adj_mats, features_labelled_ts, classes_ts = ass[35:49], fss[35:49], css[35:49]

# 0 - illicit, 1 - licit
labels_ts = []
for c in classes_ts:
    labels_ts.append(np.array(c['class'] == '2', dtype=np.long))

gcn = GCN_2layer(num_features, num_hiddens, num_classes)
gcn.load_state_dict(torch.load("gcn_weights.pth"))

# # Testing

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
    test_precisions.append(precision_score(L, test_pred))
    test_recalls.append(recall_score(L, test_pred))
    test_f1s.append(f1_score(L, test_pred))

    y_true.extend(L.tolist())
    y_pred.extend(test_pred.tolist())
#
# # # 计算混淆矩阵
# # confusion_mat = confusion_matrix(y_true, y_pred)
# # TP = confusion_mat[1, 1]
# # TN = confusion_mat[0, 0]
# # FP = confusion_mat[0, 1]
# # FN = confusion_mat[1, 0]
# #
# # print("TP:", TP)
# # print("TN:", TN)
# # print("FP:", FP)
# # print("FN:", FN)
#
acc = np.array(test_accs).mean()
prec = np.array(test_precisions).mean()
rec = np.array(test_recalls).mean()
f1 = np.array(test_f1s).mean()

print("GCN - averaged accuracy: {}, precision: {}, recall: {}, f1: {}".format(acc, prec, rec, f1))

# from sklearn.metrics import precision_score, recall_score, f1_score
#
# # Testing
# test_accs = []
# test_precisions = []
# test_recalls = []
# test_f1s = []
#
# y_true = []
# y_pred = []
#
# for ts in test_ts:
#     A = torch.tensor(adj_mats[ts].values)
#     X = torch.tensor(features_labelled_ts[ts].values)
#     L = torch.tensor(labels_ts[ts], dtype=torch.long)
#
#     gcn.eval()
#     test_out = gcn(A, X)
#
#     test_pred = test_out.max(1)[1].type_as(L)
#     t_acc = (test_pred.eq(L).double().sum()) / L.shape[0]
#     test_accs.append(t_acc.item())
#
#     y_true.extend(L.tolist())
#     y_pred.extend(test_pred.tolist())
#
# # micro
# micro_acc = np.array(test_accs).mean()
# micro_prec = precision_score(y_true, y_pred, average='micro')
# micro_rec = recall_score(y_true, y_pred, average='micro')
# micro_f1 = f1_score(y_true, y_pred, average='micro')
#
# # macro
# macro_prec = precision_score(y_true, y_pred, average='macro')
# macro_rec = recall_score(y_true, y_pred, average='macro')
# macro_f1 = f1_score(y_true, y_pred, average='macro')
#
# print("Micro - accuracy: {}, precision: {}, recall: {}, f1: {}".format(micro_acc, micro_prec, micro_rec, micro_f1))
# print("------------------------------------------------------")
# print("Macro - accuracy: {}, precision: {}, recall: {}, f1: {}".format(micro_acc, macro_prec, macro_rec, macro_f1))
