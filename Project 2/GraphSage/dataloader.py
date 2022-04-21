from collections import defaultdict
from itertools import chain

import networkx as nx
import numpy as np
import pandas as pd
from torch.utils.data import Dataset


class SocialNetworkDataset(Dataset):


def load_data(file_name):
    """
    read edges from an edge file
    """
    edges = []
    df = pd.read_csv(file_name)
    for idx, row in df.iterrows():
        user_id, friends = row["user_id"], eval(row["friends"])
        edges.extend((user_id, friend) for friend in friends)
    edges = sorted(edges)

    return edges


def load_test_data(file_name):
    """
    read edges from an edge file
    """
    scores = []
    df = pd.read_csv(file_name)
    edges = [(row["src"], row["dst"]) for idx, row in df.iterrows()]
    edges = sorted(edges)

    return edges


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