import glob
import os

import numpy as np
from matplotlib import pyplot as plt

if not os.path.exists('./plots/'):
    os.mkdir('./plots/')
np_records = glob.glob('./record_*.npy')
min_rmse = 99999
min_dim = 100
for i in np_records:
    dim = int(i.split('record_dim')[-1].split('.')[0])
    record = np.load(i, allow_pickle=True).item()
    x = list(range(1, 1 + len(record['train'])))
    plt.plot(x, record['train'], label='training')
    plt.plot(x, record['valid'], label='validation')
    plt.xlabel("Epoch")
    plt.ylabel("RMSE")
    plt.title(f"RMSE plot for embedding dimension {dim}")
    plt.legend()
    plt.savefig(f'./plots/{dim}.png', bbox_inches='tight')
    plt.clf()
    if min(record['valid']) < min_rmse:
        min_rmse = min(record['valid'])
        min_dim = dim
print(min_dim, min_rmse)
