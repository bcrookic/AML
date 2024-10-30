import pickle
import networkx as nx
from gcn.node2vec import Node2Vec


if __name__ == "__main__":
    G = nx.read_edgelist('./dataset/blockchain_sim.edgelist',
                           create_using=nx.DiGraph(),
                           nodetype=None,
                           data=[('weight', float)])

    model = Node2Vec(G, 10, 80, workers=1, p=0.25, q=2, use_rejection_sampling=0)
    model.train()
    embeddings = model.get_embeddings()
    pickle.dump(embeddings, open('data/dataset/embedding_sim.pkl', 'wb'))
