import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # 激励函数都在这
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score


# 加载数据

def load_data(data_dir, start_ts, end_ts):
    classes_csv = 'elliptic_txs_classes.csv'
    edgelist_csv = 'elliptic_txs_edgelist.csv'
    features_csv = 'elliptic_txs_features.csv'

    classes = pd.read_csv(os.path.join(data_dir, classes_csv),
                          index_col='txId')  # labels for the transactions i.e. 'unknown', '1', '2'
    edgelist = pd.read_csv(os.path.join(data_dir, edgelist_csv),
                           index_col='txId1')  # directed edges between transactions
    features = pd.read_csv(os.path.join(data_dir, features_csv), header=None,
                           index_col=0)  # features of the transactions

    num_tx = features.shape[0]
    total_tx = list(classes.index)

    # select only the transactions which are labelled
    labelled_classes = classes[classes['class'] != 'unknown']
    labelled_tx = list(labelled_classes.index)

    # to calculate a list of adjacency matrices for the different timesteps

    features_labelled_ts = []
    classes_ts = []

    for ts in range(start_ts, end_ts):
        features_ts = features[features[1] == ts + 1]
        tx_ts = list(features_ts.index)

        labelled_tx_ts = [tx for tx in tx_ts if tx in set(labelled_tx)]

        # adjacency matrix for all the transactions
        # we will only fill in the transactions of this timestep which have labels and can be used for training
        adj_mat = pd.DataFrame(np.zeros((num_tx, num_tx)), index=total_tx, columns=total_tx)

        edgelist_labelled_ts = edgelist.loc[edgelist.index.intersection(labelled_tx_ts).unique()]
        for i in range(edgelist_labelled_ts.shape[0]):
            adj_mat.loc[edgelist_labelled_ts.index[i], edgelist_labelled_ts.iloc[i]['txId2']] = 1

        features_l_ts = features.loc[labelled_tx_ts]

        features_labelled_ts.append(features_l_ts)
        classes_ts.append(classes.loc[labelled_tx_ts])

    return features_labelled_ts, classes_ts


features_labelled_ts, classes_ts = load_data(dir, 0, 49)


def get_xy(i):
    features, classes = features_labelled_ts[i], classes_ts[i]

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


net = Net(n_feature=166, n_hidden=83, n_output=2)  # 几个类别就几个 output

# 训练网络
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

# 测试网络

accuracies = []
precisions = []
recalls = []
f1_scores = []

for i in range(35, 49):
    x, y = get_xy(i)

    with torch.no_grad():
        net.eval()  # 将模型设置为评估模式
        # 进行单次测试，并获得准确率、精确率、召回率和F1值
        out = net(x)
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()

    accuracy = accuracy_score(target_y, pred_y)
    precision = precision_score(target_y, pred_y)
    recall = recall_score(target_y, pred_y)
    f1 = f1_score(target_y, pred_y)

    # 将指标值添加到列表中
    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

# 计算平均值
accuracy = np.mean(accuracies)
precision = np.mean(precisions)
recall = np.mean(recalls)
f1 = np.mean(f1_scores)

print("Averaged Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
