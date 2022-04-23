from collections import defaultdict
from itertools import chain

import networkx as nx
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


def generate_false_edges(true_edges, num_false_edges=5):
    """
    generate false edges given true edges
    """
    nodes = list(set(chain.from_iterable(true_edges)))
    N = len(nodes)
    true_edges = set(true_edges)
    print(N, len(true_edges))
    false_edges = set()

    while len(false_edges) < num_false_edges:
        # randomly sample two different nodes and check whether the pair exisit or not
        src, dst = nodes[int(np.random.rand() * N)], nodes[int(np.random.rand() * N)]
        if src != dst and (src, dst) not in true_edges and (src, dst) not in false_edges:
            false_edges.add((src, dst))
    false_edges = sorted(false_edges)

    return false_edges


def construct_graph_from_edges(edges):
    """
    generate a directed graph object given true edges
    DiGraph documentation: https://networkx.github.io/documentation/stable/reference/classes/digraph.html
    """
    # convert a list of edges {(u, v)} to a list of edges with weights {(u, v, w)}
    edge_weight = defaultdict(float)
    for e in edges:
        edge_weight[e] += 1.0
    weighed_edge_list = [(e[0], e[1], edge_weight[e]) for e in sorted(edge_weight.keys())]

    graph = nx.DiGraph()
    graph.add_weighted_edges_from(weighed_edge_list)

    print("number of nodes:", graph.number_of_nodes())
    print("number of edges:", graph.number_of_edges())

    return graph


class SocialNetworkDataset(Dataset):
    def __init__(self, total_length, mode='train', negative_sample=5):
        assert mode in ['train', 'valid', 'test']
        self.mode = mode
        self.negative_samples = negative_sample
        self.edges = []
        data = pd.read_csv(f'./data/{mode}', index_col=None)
        for idx, row in data.iterrows():
            user_id, friends = row["user_id"], eval(row["friends"])
            self.edges.extend((user_id, friend) for friend in friends)
        self.edges = sorted(self.edges)
        if total_length:
            self.negative_samples = generate_false_edges(self.edges, num_false_edges=total_length - len(self.edges))
        elif negative_sample == 'same':
            self.negative_samples = generate_false_edges(self.edges, num_false_edges=len(self.edges))
        elif type(negative_sample) is int:
            self.negative_samples = generate_false_edges(self.edges, num_false_edges=negative_sample)

        self.total_edges = self.edges + self.negative_samples

        self.graph = construct_graph_from_edges(self.edges)
        self.total_graph = construct_graph_from_edges(self.total_edges)

    def __len__(self):
        return len(self.total_edges)

    def get_positive_edges(self):
        return self.edges

    def get_total_edges(self):
        return self.total_edges

    def get_total_graph(self):
        return self.total_graph

    def get_positive_graph(self):
        return self.graph

    def __getitem__(self, item):
        return self.total_edges[item]
