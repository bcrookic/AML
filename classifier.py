import tensorflow as tf
import numpy as np
import random
import argparse
import pickle
import csv


def batch_iter(data, batch_size, num_epochs, shuffle=True):
    """
    Generates a batch iterator for a dataset.
    为数据集生成一个批迭代器。
    data：数据集，可以是任何类型的序列。
    batch_size：一个批次的大小，即每个批次中包含的数据项数。
    num_epochs：训练数据的轮数。
    shuffle：一个布尔值，表示是否对数据集进行随机排序（打乱顺序）。
    如果设置了shuffle为True，那么就对数据进行打乱，即使用np.random.permutation函数对数据集的下标进行随机排序，然后根据这个排序结果重新排列数据集。
    """
    data = np.asarray(data)  # 将输入数据data转换为NumPy数组
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
    '''
    每个epoch中，使用一个嵌套的循环迭代数据集的每个批次。
    对于每个批次，计算其起始索引start_index和结束索引end_index，
    然后使用切片操作从打乱的数据集中提取该批次的数据，并使用yield关键字将其返回。
    yield关键字将函数转换为一个生成器，它能够暂停函数执行并返回结果，然后在需要时恢复函数执行。
    '''


def str2float(str):
    def is_num(char):
        """
        检查字符是否是数字、小数点、负号、指数符号等有效字符
        如果字符在该列表中，则返回True，否则返回False。
        """
        return char in ['0', '1', '2', '3', '4', '5', '6', '7', '8', '9', '.', '-', 'e']

    tmp = ''.join(list(filter(is_num, str)))  # 提取数字和其他有效字符
    return 0 if len(tmp) < 1 else tmp  # 如果tmp的长度小于1，则函数返回0，否则将tmp转换为浮点数并返回。



def getData(mode, emb_file, feature_file, label_file, mapping_file):
    '''
    目的是从参数指定的文件中读入数据，处理数据，并以适合用于训练和测试机器学习模型的格式返回数据.
    mode:指定要使用的功能。可能的值为“AF”、“LF”、“NE”、“NE+AF”和“NE+LF”。
    emb_file:包含网络嵌入的二进制文件的路径。
    feature_file:包含网络中每个节点要素的 CSV 文件的路径。
    label_file:包含网络中每个节点的标签的文件的路径。
    mapping_file:包含节点 ID 和索引之间映射的文件的路径。
    '''
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
    '''
    标签、特征、id2index映射、嵌入、正样本和负样本、打乱后的训练数据和测试数据
    '''

    # get labels
    '''
    读取label_file文件中的标签数据，将其转换为整数类型并添加到label列表中。
    其中strip()方法用于去掉字符串两端的空白字符，
    end='\r'表示输出不换行，而是回车到行首。这样可以实现输出进度条的效果。
    '''
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
    '''
    获取特征数据：
    根据mapping_file中的内容，将id映射到index，存储在id2index中；
    预先定义一个空列表feature，长度为标签列表label的长度，用于存储特征数据；
    打开特征文件feature_file，并读取其中的数据；
    对于每一行数据，根据第一列中的id，在id2index中查找对应的index，并将该行的特征数据转化为浮点数，并存储在feature的相应位置上。
    '''
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
    '''
    获取网络嵌入数据：
    打开网络嵌入文件emb_file，并读取其中的数据；
    预先定义一个空列表emb，长度为网络嵌入数据的长度；
    遍历网络嵌入数据，将每个key所对应的value赋值给emb列表的相应位置上。
    '''
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
    '''
    将特征和网络嵌入数据组合：
    如果是"AF"、"LF"、"NE+AF"、"NE+LF"中的任意一种，则将特征数据和网络嵌入数据组合在一起，形成新的特征向量。对于每一个样本，如果其标签为0，则将其特征向量存储在neg_sample列表中；如果其标签为1，则将其特征向量存储在pos_sample列表中；
    如果是"NE"中的一种，则直接使用网络嵌入数据作为样本数据。对于每一个样本，如果其标签为0，则将其网络嵌入数据存储在neg_sample列表中；如果其标签为1，则将其网络嵌入数据存储在pos_sample列表中；
    pos_sample和neg_sample列表分别存储了所有的正样本和负样本的特征向量或网络嵌入数据；
    emb_size记录了特征向量或网络嵌入数据的长度。
    '''
    if mode == "AF" or mode == "LF":
        for i in range(len(emb)):
            if label[i] == 0:
                neg_sample.append(feature[i])
            elif label[i] == 1:
                pos_sample.append(feature[i])
    elif mode == "NE":
        # while len(emb) < len(label):
        #     emb.append([])
        for i in range(len(emb)):
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
    '''
    获取训练集和测试集：
    对于pos_sample和neg_sample，分别进行随机打乱；
    根据8:2的正样本和负样本比例，将pos_sample和neg_sample划分为训练集和测试集，
    train_data存储了训练集的所有样本数据，test_data存储了测试集的所有样本数据；
    '''
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
    '''
    创建神经网络层:
    inputs 是输入数据，即这一层神经网络的输入；
    in_size 是输入数据的大小，即输入数据的特征数量；
    out_size 是输出数据的大小，即这一层神经网络的输出特征数量；
    activation_function 是激活函数，用于在神经网络中对输出进行非线性变换。
    '''
    Weights = tf.Variable(tf.truncated_normal([in_size,out_size],mean=0, stddev=0.2))
    biases = tf.Variable(tf.zeros([1,out_size]) + 0.1)
    '''
    Weights 和 biases 分别表示这一层神经网络的权重和偏差。
    Weights 是一个形状为 [in_size, out_size] 的张量，
    其中 in_size 表示输入数据的特征数量，out_size 表示输出数据的特征数量。这个张量的值是从截断正态分布中随机采样得到的，均值为 0，标准差为 0.2
    biases 是一个形状为 [1, out_size] 的张量，
    其中第一个维度为 1，第二个维度为输出数据的特征数量。这个张量的值被初始化为一个全零张量，并且每个元素都加上了 0.1 的偏差。
    '''
    Wx_plus_b = tf.matmul(inputs, Weights) + biases
    '''
    函数使用 tf.matmul 函数计算输入数据 inputs 和权重 Weights 的矩阵乘积，
    然后加上偏差 biases，得到一个形状为 [batch_size, out_size] 的张量 Wx_plus_b。
    其中，batch_size 表示输入数据的批次大小，即一次输入的数据量。
    '''
    if activation_function is None:
        outputs = Wx_plus_b
    else:
        outputs = activation_function(Wx_plus_b)
    return outputs


def to_list(arr):
    if isinstance(arr, np.ndarray):
        return arr.tolist()
    elif isinstance(arr, (list, tuple)):
        return [to_list(x) for x in arr]
    else:
        return arr









parser = argparse.ArgumentParser()  # 从命令行中读取参数
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
emb_file = "../data/dataset/embedding_sim.pkl"
feature_file = "./raw_data/elliptic_txs_features.csv"
label_file = "../data/dataset/label.txt"
mapping_file = "../data/dataset/id2index.txt"
emb_size, train_data_shuffled, train_label_shuffled, test_data_shuffled, test_label_shuffled = getData(mode,
                                                                                                       emb_file,
                                                                                                       feature_file,
                                                                                                       label_file,
                                                                                                       mapping_file)


#Multilayer Perceptron
x = tf.placeholder(tf.float32, [None, emb_size])
y = tf.placeholder(tf.int64, [None])
keep_prob = tf.placeholder(tf.float32)
'''
x是一个浮点型张量，形状为[None, emb_size]，表示输入的特征，None表示可以接受任意数量的输入。
y是一个整型张量，形状为[None]，表示对应每个输入的标签。
keep_prob是一个浮点型张量，用于控制随机失活的比例。
'''
l1 = add_layer(x, emb_size, mlp_hidden, activation_function=tf.nn.relu)
l1_dropout = tf.nn.dropout(l1,keep_prob)
prob = add_layer(l1_dropout, mlp_hidden, 2, activation_function=tf.nn.softmax)
'''
add_layer函数用于添加一层全连接层，并指定激活函数。
l1表示第一层隐藏层的输出，形状为[None, mlp_hidden]。
l1_dropout表示在第一层之后添加的随机失活层，以防止过拟合。
prob表示输出层的输出，形状为[None, 2]，每一行代表一个输入对应两个类别的概率。
'''
prediction = tf.argmax(prob, 1)
whether_correct = tf.equal(y, prediction)
accuracy = tf.reduce_mean(tf.cast(whether_correct, tf.float32))
loss = tf.reduce_mean(tf.reduce_sum((prob - tf.one_hot(y, 2))**2, reduction_indices=[1]))
'''
prediction表示模型的预测结果，即在输出层选择概率最大的类别。
whether_correct表示每个样本的预测是否正确。
accuracy表示模型的准确率。
loss表示模型的损失函数，使用平方误差损失函数。
'''


#training...
train_step = tf.train.AdagradOptimizer(lr).minimize(loss) # 优化损失函数
#train_step = tf.train.AdamOptimizer(lr).minimize(loss)
init = tf.global_variables_initializer()  # 初始化 TensorFlow 的全局变量
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

    _, pre_list, l, acc= sess.run([train_step, prediction, loss, accuracy], feed_dict={x:x_batch, y:y_batch})  # ValueError: setting an array element with a sequence.
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
