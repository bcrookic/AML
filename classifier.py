import tensorflow as tf
import numpy as np
import random
import argparse
from sklearn import svm
import pickle
import csv


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    """
    data = np.asarray(data)
    data_size = len(data)
    num_batches_per_epoch = int(len(data) / batch_size) + 1
    for epoch in range(num_epochs):
        # Shuffle the data at each epoch
        if shuffle:
            shuffle_indices = np.random.permutation(np.arange(data_size))
            shuffled_data = data[shuffle_indices]
        else:
            shuffled_data = data
        for batch_num in range(num_batches_per_epoch):
            start_index = batch_num * batch_size
            end_index = min((batch_num + 1) * batch_size, data_size)
            yield shuffled_data[start_index:end_index]


def str2float(str):
    def is_num(char):
        return char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', 'e']

    tmp = ''.join(list(filter(is_num, str)))
    return 0 if len(tmp) < 1 else tmp


def getData(mode, emb_file, feature_file, label_file, mapping_file):
    label = []
    feature = []
    id2index = {}
    emb = []
    pos_sample = []
    neg_sample = []
    train_data_shuffled = []
    train_label_shuffled = []
    test_data_shuffled = []
    test_label_shuffled = []

    # get labels
    with open(label_file, 'r') as f:
        lines = f.readlines()
        i = 0
        for l in lines:
            l = l.strip()
            label.append(int(l))
            i += 1
            print("Processing label.txt data: %d" % i, end='\r')
        print()

    # get features
    if mode == "AF" or mode == "LF" or mode == "NE+LF" or mode == "NE+AF":
        with open(mapping_file, 'r') as f:
            lines = f.readlines()
            i = 0
            for l in lines:
                i += 1
                l = l.strip().split('\t')
                id2index[l[0]] = int(l[1])
                print("Processing id2index.txt data: %d" % i, end='\r')
            print()
        feature = [[] for _ in range(len(label))]
        with open(feature_file) as f:
            f_csv = csv.reader(f)
            i = 0
            for row in f_csv:
                i += 1
                tmp = []
                for num in row[1:]:
                    tmp.append(float(str2float(num)))
                if mode == "AF" or mode == "NE+AF":
                    feature[id2index[row[0]]] = tmp
                elif mode == "LF" or mode == "NE+LF":
                    feature[id2index[row[0]]] = tmp[:94]
                print("Processing features.csv data: %d" % i, end='\r')
            print()
        print("Feature num: ", len(feature))

    # get network embeddings
    if mode == "NE" or mode == "NE+LF" or mode == "NE+AF":
        with open(emb_file, 'rb') as f:
            raw_data = pickle.load(f)
            emb = [[] for _ in range(len(raw_data))]
            i = 0
            for key, value in raw_data.items():
                emb[int(key)] = value
                i += 1
                print("Processing emb data: %d" % i, end='\r')
            print()

    # input data
    if mode == "AF" or mode == "LF":
        for i in range(len(label)):
            if label[i] == 0:
                neg_sample.append(feature[i])
            elif label[i] == 1:
                pos_sample.append(feature[i])
    elif mode == "NE":
        for i in range(len(label)):
            if label[i] == 0:
                neg_sample.append(emb[i])
            elif label[i] == 1:
                pos_sample.append(emb[i])
    elif mode == "NE+LF" or mode == "NE+AF":
        for i in range(len(label)):
            if label[i] == 0:
                neg_sample.append(list(emb[i]) + list(feature[i]))
            elif label[i] == 1:
                pos_sample.append(list(emb[i]) + list(feature[i]))
    print("Positive sample:", len(pos_sample))
    print("Negative sample:", len(neg_sample))
    emb_size = len(pos_sample[-1])
    print("Embedding size: ", emb_size)


    # get train&test dataset
    random.shuffle(pos_sample)
    random.shuffle(neg_sample)
    pos_index = int(len(pos_sample) * 0.8)
    neg_index = int(len(neg_sample) * 0.8)
    train_data = pos_sample[:pos_index] + neg_sample[:neg_index]
    train_label = [1 for _ in range(pos_index)] + [0 for _ in range(neg_index)]
    test_data = pos_sample[pos_index:] + neg_sample[neg_index:]
    test_label = [1 for _ in range(len(pos_sample) - pos_index)] + [0 for _ in range(len(neg_sample) - neg_index)]
    print("Training data: {:g}, Testing data: {:g}".format(len(train_data), len(test_data)))
    tmp = [i for i in zip(train_data, train_label)]
    random.shuffle(tmp)
    for i, j in tmp:
        train_data_shuffled.append(i)
        train_label_shuffled.append(j)
    tmp = [i for i in zip(test_data, test_label)]
    random.shuffle(tmp)
    for i, j in tmp:
        test_data_shuffled.append(i)
        test_label_shuffled.append(j)
    return emb_size, train_data_shuffled, train_label_shuffled, test_data_shuffled, test_label_shuffled


def add_layer(inputs, in_size, out_size, activation_function=None):
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size],mean=0, stddev=0.2))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


parser = argparse.ArgumentParser()
parser.add_argument('--mlp_hidden', type=int, default=32)    #32, 64, 100
parser.add_argument('--lr', type=float, default=0.05)
parser.add_argument('--mode', type=str, default="NE")    #NE, AF, LF, NE+AF, NE+LF
parser.add_argument('--lm', type=float, default=0.70)
args = parser.parse_args()


mlp_hidden = args.mlp_hidden
lr = args.lr
num_epoch = 500
batch_size = 64
mode = args.mode
limit = args.lm
emb_file = "./dataset/embedding_sim.pkl"
feature_file = "./raw_data/elliptic_txs_features.csv"
label_file = "./dataset/label.txt"
mapping_file = "./dataset/id2index.txt"
emb_size, train_data_shuffled, train_label_shuffled, test_data_shuffled, test_label_shuffled = getData(mode,
                                                                                                       emb_file,
                                                                                                       feature_file,
                                                                                                       label_file,
                                                                                                       mapping_file)


#Multilayer Perceptron
x = tf.placeholder(tf.float32, [None, emb_size])
y = tf.placeholder(tf.int64, [None])
keep_prob = tf.placeholder(tf.float32)
l1 = add_layer(x, emb_size, mlp_hidden, activation_function=tf.nn.relu)
l1_dropout = tf.nn.dropout(l1,keep_prob)
prob = add_layer(l1_dropout, mlp_hidden, 2, activation_function=tf.nn.softmax)
prediction = tf.argmax(prob, 1)
whether_correct = tf.equal(y, prediction)
accuracy = tf.reduce_mean(tf.cast(whether_correct, tf.float32))
loss = tf.reduce_mean(tf.reduce_sum((prob - tf.one_hot(y, 2))**2, reduction_indices=[1]))

#######################################
#Cross entropy loss
#losses = tf.nn.softmax_cross_entropy_with_logits(logits=prob, labels=tf.one_hot(y, 2))
#losses = tf.nn.sigmoid_cross_entropy_with_logits(logits=prob, labels=tf.one_hot(y, 2))
#loss = -tf.reduce_mean(losses)
#######################################

######################################################
#Weighted cross entropy loss
#losses = tf.nn.weighted_cross_entropy_with_logits(
#             targets=tf.one_hot(y, 2),
#             logits=prob,
#             pos_weight=tf.constant(float(0.7/0.3)),
#             name=None
#         )
#loss = -tf.reduce_mean(losses)
#######################################################

#######################################
#Focal loss
#cross_entropy = tf.multiply(tf.log(tf.clip_by_value(prob,1e-5,1.0)), tf.one_hot(y, 2))
#weight = tf.pow(tf.subtract(1.0,prob), 10)   # n = 次方
#loss = -tf.reduce_mean(tf.multiply(cross_entropy, weight))
#######################################

#######################################
#Huber loss
#delta=1.0
#residual = tf.abs(prob - tf.one_hot(y, 2))
#condition = tf.less(residual, delta) # delta=1.0
#small_res = 0.5 * tf.square(residual)
#large_res = delta * residual - 0.5 * tf.square(delta)
#loss = -tf.reduce_mean(tf.where(condition, small_res, large_res)）
#######################################


#training...
train_step = tf.train.AdagradOptimizer(lr).minimize(loss)
#train_step = tf.train.AdamOptimizer(lr).minimize(loss)
init = tf.global_variables_initializer()
sess = tf.Session()
sess.run(init)
batches = batch_iter(list(zip(train_data_shuffled, train_label_shuffled)), batch_size, num_epoch)
sum_batches = len(train_data_shuffled)//batch_size
p = 0
r = 0
f_score = 0
for i in range(sum_batches * num_epoch):
    batch = batches.__next__()
    if len(batch)<1:
        continue
    x_batch, y_batch = zip(*batch)
    if len(y_batch)<1:
        continue
    _, pre_list, l, acc= sess.run([train_step, prediction, loss, accuracy], feed_dict={x:x_batch, y:y_batch, keep_prob:0.75})
    #print(pre_list)
    if i % 100 == 0:
        print("TRAIN: step {}, loss {:g}, acc {:g}".format(i, l, acc))

    if i % sum_batches == 0:
        j = 0
        tp = 0
        fp = 0
        fn = 0
        tn = 0
        precision = 0.0
        recal = 0.0
        f_score = 0.0
        ctr_cor = 0
        ctr_sum = 0
        acc_sum = 0
        l_sum = 0
        test_sum_batches = 0
        test_batches = batch_iter(list(zip(test_data_shuffled, test_label_shuffled)), batch_size, 1)
        for tb in test_batches:
            if len(tb)<1:
                continue
            test_x_batch, test_y_batch = zip(*tb)
            if len(test_y_batch)<1:
                continue
            pre_list, acc, l = sess.run([prediction, accuracy, loss], feed_dict={x:test_x_batch, y:test_y_batch, keep_prob:1.0})
            l_sum += l
            acc_sum += acc
            test_sum_batches += 1
            start = j * batch_size
            end = min((j + 1) * batch_size, len(test_label_shuffled))
            for k in range(len(test_y_batch)):
                if test_y_batch[k] == 1 and pre_list[k] == 1:
                    tp += 1
                if test_y_batch[k] == 0 and pre_list[k] == 1:
                    fp += 1
                if test_y_batch[k] == 1 and pre_list[k] == 0:
                    fn += 1
                if test_y_batch[k] == 0 and pre_list[k] == 0:
                    tn += 1
            j += 1
        assert(len(test_label_shuffled)==(tp+fp+fn+tn))
        if tp+fp != 0:
            p = tp/(tp+fp)
        if tp+fn != 0:
            r = tp/(tp+fn)
        if p+r != 0:
            f_score = 2*p*r / (p+r)

        if f_score >= limit:
            with open("res_mlp_"+mode+str(limit)+".txt", 'a+') as f:
                f.write(str('%.3f'%p)+'\t'+str('%.3f'%r)+'\t'+str('%.3f'%f_score)+'\t'+str(mlp_hidden)+'\t'+str(lr)+'\n')
        print("DEV: epoch {}, loss {:g}, acc {:g}, Precision {:g}, Recall {:g}, F-score {:g}".format(int(i/sum_batches), l_sum/test_sum_batches, acc_sum/test_sum_batches, p, r, f_score))
