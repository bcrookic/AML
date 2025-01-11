import networkx as nx
import pickle
import csv
import re
import time
from queue import Queue


def calculateAccSim(index_max, edges):
    INT_MAX = float("inf")
    dis_thre = 10
    accsim = {}
    ticks = time.asctime(time.localtime(time.time()))
    print("Start:", ticks)
    for i in range(index_max):
        que = Queue(maxsize=0)
        Accstep = [INT_MAX for _ in range(index_max)]
        visited = [0 for _ in range(index_max)]
        step = 0

        que.put((i, step))
        while not que.empty():
            j, step = que.get()
            if visited[j]:
                continue
            if step >= dis_thre:
                break
            if i == j:
                Accstep[j] = 0
                visited[j] = 1
                accsim[(i, j)] = 1
            else:
                if step < Accstep[j]:
                    Accstep[j] = step
                    visited[j] = 1
                    accsim[(i, j)] = 1/Accstep[j]
            for k in range(index_max):
                if (j, k) in edges and visited[k] == 0:
                    que.put((k, step+1))
        print("Calculating accessibility similarity matrix: %d" % i, end='\r')
    ticks = time.asctime(time.localtime(time.time()))
    print("Down:", ticks)
    #print(accsim)
    return accsim


def getEdges(edges_file):
    id2index = {}
    weight = {}
    edges_w = []
    edges_sim = []
    with open(edges_file) as f:
        f_csv = csv.reader(f)
        i, index = 0, 0
        for row in f_csv:
            i += 1
            if i == 1:
                continue
            if row[0] not in id2index.keys():
                id2index[row[0]] = index
                index += 1
            if row[1] not in id2index.keys():
                id2index[row[1]] = index
                index += 1
            if (id2index[row[0]], id2index[row[1]]) not in weight.keys():
                weight[(id2index[row[0]], id2index[row[1]])] = 1
            else:
                weight[(id2index[row[0]], id2index[row[1]])] += 1
            print("Processing elliptic_txs_edgelist.csv data: %d" % i, end='\r')
    print()
    print("Nodes: ", index+1)
    print("Edges: ", len(weight))

    index_max = index
    node = [i for i in range(index_max)]
    for k, v in weight.items():
        edges_w.append((k[0], k[1], v))
    acc_sim = calculateAccSim(index_max, weight)
    for k, v in acc_sim.items():
        if k[0] != k[1]:
            edges_sim.append((k[0], k[1], v))
    return id2index, node, edges_w, edges_sim


def getLabel(index_max, id2index, label_file):
    unknow = 0
    illicit = 0
    licit = 0
    label = [0 for _ in range(index_max)]
    with open(label_file) as f:
        f_csv = csv.reader(f)
        i = 0
        for row in f_csv:
            i += 1
            if i == 1:
                continue
            if row[1] == 'unknown':
                label[id2index[row[0]]] = 2
                unknow += 1
            elif row[1] == '1':
                label[id2index[row[0]]] = 1
                illicit += 1
            elif row[1] == '2':
                label[id2index[row[0]]] = 0
                licit += 1
            print("Processing elliptic_txs_classes.csv data: %d" % i, end='\r')
    print()
    print("Unknow: ", unknow)
    print("Illicit: ", illicit)
    print("Licit: ", licit)
    return label


def saveMap(id2index, save_file):
    with open(save_file, 'a+') as f:
        for k, v in id2index.items():
            f.write(str(k) + '\t' + str(v) + '\n')


def saveGraph(nodes, edges, save_file):
    G = nx.DiGraph()
    G.add_nodes_from(nodes)
    G.add_weighted_edges_from(edges)
    #nx.write_gpickle(G, save_file)    #save in pkl
    nx.write_weighted_edgelist(G, save_file, delimiter=' ')    #save in edgelist


def saveLabel(label, save_file):
    with open(save_file, 'a+') as f:
        for i in label:
            f.write(str(i) + '\n')


if __name__ == "__main__":
    edges_file = "./raw_data/elliptic_txs_edgelist.csv"
    label_file = "./raw_data/elliptic_txs_classes.csv"
    save_file = "./dataset/"

    id2index, nodes, edges_w, edges_sim = getEdges(edges_file)
    label = getLabel(len(nodes), id2index, label_file)

    saveMap(id2index, save_file+"id2index.txt")
    saveGraph(nodes, edges_w, save_file+"blockchain.edgelist")
    saveGraph(nodes, edges_sim, save_file+"blockchain_sim.edgelist")
    saveLabel(label, save_file+"label.txt")

    print(nodes[-1])
    print(edges_w[-1])
    print(edges_sim[-1])
