# Paper:
A Plug-and-Play Data-Driven Approach for Anti-Money Laundering in Bitcoin


# Requirements
 - tensorflow
 - python >= 3.6
 - numpy
 - networkx 
 - torch 
 - gensim
 - sklearn

# Prepare Data
Source code and data for GCN-elliptic dataset (https://www.kaggle.com/code/karthikapv/gcn-elliptic-dataset/notebook)


Due to file size restrictions on uploads, before running the models, some preprocessing is needed:
Download data from `https://pan.quark.cn/s/a7b2b764f84c`
and put them into corresponding folders


# How to run
1. Use preprocessor.py to get graph（based on similarity or not) as the input of graph embedding model
    
    `command: python preprocessor.py`

   You can also directly obtain relevant files from the existing dataset folder

2. Graph embedding model include：

   main.py

   nodeEmb.py

   walker.py

   alias.py

   utils.py
    
    We use `networkx`to create graphs.The input of networkx graph is as follows:
    entity1 entity2 weight/similarity
  
    `command: python main.py`

4. Dimension reduction of the elliptic data:  

    Use gcn_get_pkl.py to get the `.pkl` of the elliptic data

    `command: python gcn_get_pkl.py`

    Use Ordered_dictionary.py to get the ordered embedding_sim of the embedding

    `command: python Ordered_dictionary.py`

    Use TruncatedSVD.py to reduce the dimensionality of the data

    `command: python TruncatedSVD.py`

    You can merge the generated dimensionality reduction feature data and native elliptic feature data into a new feature file (`.csv`)

    `command: python pkl_csv.py`
    

5. Binary classification model include：
    classifier.py

   `command: python classifier.py --mlp_hidden=64 --lr=0.05 --mode=NE+AF --lm=0.7`

# Wait to do
1. Stability over the training percentage of the graph embedding
2. Stability over the length of random walk on graph
3. Stability over the predefined distance threshold in similarity calculating





