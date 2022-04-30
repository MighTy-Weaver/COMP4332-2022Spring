import numpy as np
from gensim.models import Word2Vec, KeyedVectors
from tqdm import tqdm

from dataloader import SocialNetworkDataset

model = KeyedVectors.load_word2vec_format('../node2vec_best_model.bin', binary=True)
print(len(model.vectors))

ds_train = SocialNetworkDataset(mode='train', negative_sample='same')
ds_val = SocialNetworkDataset(mode='valid', negative_sample='same')
# ds_neg = SocialNetworkDataset(mode='test', negative_sample='same')

embedding_dict = {}
for ds in [ds_train, ds_val]:
    nodes = list(ds.graph.nodes)
    edges = list(ds.graph.edges)
    edge = edges[0]
    print(edge[0], edge[1])
    for n in tqdm(nodes):
        try:
            embedding_dict[n] = model[n]
        except KeyError:
            embedding_dict[n] = np.random.rand(10)

np.save('./embedding.npy', embedding_dict)
