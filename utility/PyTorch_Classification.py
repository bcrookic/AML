import os
import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  
from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score



def load_data(data_dir, start_ts, end_ts):
    classes_csv = 'elliptic_txs_classes.csv'
    edgelist_csv = 'elliptic_txs_edgelist.csv'
    features_csv = 'new_features.csv'

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

dir = "./raw_data"
features_labelled_ts, classes_ts = load_data(dir, 0, 49)


def get_xy(i):
    features, classes = features_labelled_ts[i], classes_ts[i]

    x_tensor_list = [torch.tensor(features[col].astype(float).values, dtype=torch.float32) for col in features.columns]
    features = torch.stack(x_tensor_list, dim=1) 

    classes = torch.tensor(classes['class'].astype(int).values - 1, dtype=torch.long) 

    return features, classes


class Net(torch.nn.Module): 
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__() 
        self.hidden = torch.nn.Linear(n_feature, n_hidden)  
        self.out = torch.nn.Linear(n_hidden, n_output)  

    def forward(self, x):
        x = F.relu(self.hidden(x))  
        x = self.out(x)  
        return x

num_features = 166 + 64  # 166 + TruncatedSVD_dimensions
num_hiddens = num_features // 2  # Generally, num_features divided by 2

net = Net(n_feature=num_features, n_hidden=num_hiddens, n_output=2)  

optimizer = torch.optim.SGD(net.parameters(), lr=0.02)
loss_func = torch.nn.CrossEntropyLoss()

for i in range(0, 35):
    x, y = get_xy(i)

    for t in range(100):
        out = net(x)  

        loss = loss_func(out, y)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()



accuracies = []
precisions = []
recalls = []
f1_scores = []

for i in range(35, 49):
    x, y = get_xy(i)

    with torch.no_grad():
        net.eval()  
        out = net(x)
        prediction = torch.max(F.softmax(out), 1)[1]
        pred_y = prediction.data.numpy().squeeze()
        target_y = y.data.numpy()

    accuracy = accuracy_score(target_y, pred_y)
    precision = precision_score(target_y, pred_y)
    recall = recall_score(target_y, pred_y)
    f1 = f1_score(target_y, pred_y)

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)

accuracy = np.mean(accuracies)
precision = np.mean(precisions)
recall = np.mean(recalls)
f1 = np.mean(f1_scores)

print("Averaged Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
