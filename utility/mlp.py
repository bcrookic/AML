import pickle

import numpy as np
import pandas as pd
import torch
import torch.nn.functional as F  # Activation function

path = './raw_data_pkl(+re)/features_labelled_ts.pkl'
f = open(path, 'rb')
fss = pickle.load(f)

path = './raw_data_pkl(+re)/classes_ts.pkl' 
f = open(path, 'rb')
css = pickle.load(f)




def get_xy(i):
    features, classes = fss[i], css[i]

    # Convert the string in the x_DataFrame column to a tensor
    x_tensor_list = [torch.tensor(features[col].astype(float).values, dtype=torch.float32) for col in features.columns]
    features = torch.stack(x_tensor_list, dim=1)  # 形状为 (x, y)，其中 x 是行数，y 是列数

    # Convert the string in y_DataFrame to a tensor of type long and perform value mapping
    classes = torch.tensor(classes['class'].astype(int).values - 1, dtype=torch.long)  # 形状为 (x,)

    return features, classes


# Establish a neural network
class Net(torch.nn.Module):  # Inherit Torch's Module
    def __init__(self, n_feature, n_hidden, n_output):
        super(Net, self).__init__()  
        self.hidden = torch.nn.Linear(n_feature, n_hidden) 
        self.out = torch.nn.Linear(n_hidden, n_output) 

    def forward(self, x):
        x = F.relu(self.hidden(x)) 
        x = self.out(x)  
        return x


f = fss[0].shape[1]
h = f // 2
net = Net(n_feature=f, n_hidden=h, n_output=2)  # Only a few categories have a few outputs

# train

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


from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score

# testing

accuracies = []
precisions = []
recalls = []
f1_scores = []
# tp_list = []
# tn_list = []
# fp_list = []
# fn_list = []

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

    # tp = np.sum(np.logical_and(pred_y == 1, target_y == 1))
    # tn = np.sum(np.logical_and(pred_y == 0, target_y == 0))
    # fp = np.sum(np.logical_and(pred_y == 1, target_y == 0))
    # fn = np.sum(np.logical_and(pred_y == 0, target_y == 1))

    accuracies.append(accuracy)
    precisions.append(precision)
    recalls.append(recall)
    f1_scores.append(f1)
    # tp_list.append(tp)
    # tn_list.append(tn)
    # fp_list.append(fp)
    # fn_list.append(fn)

accuracy = np.mean(accuracies)
precision = np.mean(precisions)
recall = np.mean(recalls)
f1 = np.mean(f1_scores)
# tp = np.sum(tp_list)
# tn = np.sum(tn_list)
# fp = np.sum(fp_list)
# fn = np.sum(fn_list)

print("Averaged Accuracy:", accuracy)
print("Precision:", precision)
print("Recall:", recall)
print("F1:", f1)
# print("TP:", tp)
# print("TN:", tn)
# print("FP:", fp)
# print("FN:", fn)



# from sklearn.metrics import accuracy_score, precision_score, recall_score, f1_score
# import numpy as np

# accuracies = []
# precisions_micro = []
# recalls_micro = []
# f1_scores_micro = []
# precisions_macro = []
# recalls_macro = []
# f1_scores_macro = []

# for i in range(35, 49):
#     x, y = get_xy(i)

#     with torch.no_grad():
#         net.eval()
#         out = net(x)
#         prediction = torch.max(F.softmax(out), 1)[1]
#         pred_y = prediction.data.numpy().squeeze()
#         target_y = y.data.numpy()

#     accuracy = accuracy_score(target_y, pred_y)
#     precision_micro = precision_score(target_y, pred_y, average='micro')
#     recall_micro = recall_score(target_y, pred_y, average='micro')
#     f1_micro = f1_score(target_y, pred_y, average='micro')
#     precision_macro = precision_score(target_y, pred_y, average='macro')
#     recall_macro = recall_score(target_y, pred_y, average='macro')
#     f1_macro = f1_score(target_y, pred_y, average='macro')

#     accuracies.append(accuracy)
#     precisions_micro.append(precision_micro)
#     recalls_micro.append(recall_micro)
#     f1_scores_micro.append(f1_micro)
#     precisions_macro.append(precision_macro)
#     recalls_macro.append(recall_macro)
#     f1_scores_macro.append(f1_macro)

# accuracy = np.mean(accuracies)
# precision_micro = np.mean(precisions_micro)
# recall_micro = np.mean(recalls_micro)
# f1_micro = np.mean(f1_scores_micro)
# precision_macro = np.mean(precisions_macro)
# recall_macro = np.mean(recalls_macro)
# f1_macro = np.mean(f1_scores_macro)

# print("Averaged Accuracy:", accuracy)
# print("Micro Precision:", precision_micro)
# print("Micro Recall:", recall_micro)
# print("Micro F1:", f1_micro)
# print("---------------------------------")
# print("Averaged Accuracy:", accuracy)
# print("Macro Precision:", precision_macro)
# print("Macro Recall:", recall_macro)
# print("Macro F1:", f1_macro)
