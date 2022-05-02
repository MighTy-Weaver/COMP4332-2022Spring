from itertools import combinations

import numpy as np
import os

from matplotlib import pyplot as plt

if not os.path.exists('./node2vec_plot/'):
    os.mkdir('./node2vec_plot/')

node_dim_choices = [5, 7, 8, 9, 10, 11, 12, 13, 15, 20, 30, 40, 50]
num_walks_choices = [5, 8, 10, 15, 20, 30]
walk_length_choices = [5, 8, 10, 15, 20, 30]
p_choices = [0.25, 0.5, 0.75, 1]
q_choices = [0.25, 0.5, 0.75, 1]

index_dict = {0: node_dim_choices, 1: num_walks_choices, 2: walk_length_choices, 3: p_choices, 4: q_choices}
range_dict = {0: 'node_dim', 1: 'num_walks', 2: 'walk_length', 3: 'p', 4: 'q'}

record = dict(np.load('./node2vec_result_dict.npy', allow_pickle=True).item())

maximum_auc = max(record.values())

for i in range(0, 5):
    # print([[record[j] for j in record.keys() if j[i] == c] for c in index_dict[i]])
    try:
        best_values = [max([record[j] for j in record.keys() if j[i] == c]) for c in index_dict[i]]
        print(range_dict[i], best_values)
        plt.plot(index_dict[i], best_values)
        plt.xlabel("Choices of {}\nHighest AUC score: {}".format(range_dict[i], round(max(best_values), 5)))
        plt.ylabel("Best AUC score achieved")
        plt.title("Highest AUC score achieved with different settings of {}".format(range_dict[i]))
        plt.savefig('./node2vec_plot/{}.png'.format(range_dict[i]), bbox_inches='tight')
        plt.clf()
    except ValueError:
        pass

for i1, i2 in combinations(range(0, 5), 2):
    bst_matrix = [[max([record[j] for j in record.keys() if j[i1] == c1 and j[i2] == c2]) for c1 in index_dict[i1]] for
                  c2 in index_dict[i2]]
    matrix = np.array(bst_matrix)
    print(matrix.shape)
    plt.imshow(matrix, cmap='hot', interpolation='nearest')
    plt.xlabel(range_dict[i1])
    plt.xticks(ticks=range(len(index_dict[i1])), labels=index_dict[i1])
    plt.ylabel(range_dict[i2])
    plt.yticks(ticks=range(len(index_dict[i2])), labels=index_dict[i2])
    plt.colorbar()
    plt.savefig('./node2vec_plot/HEAT_{}_AND_{}.png'.format(range_dict[i1], range_dict[i2]), bbox_inches='tight')
    plt.clf()
