import glob
import os

import numpy as np
import pandas as pd
from matplotlib import pyplot as plt
from tqdm import trange

if not os.path.exists('./plots/'):
    os.mkdir('./plots/')
np_records = glob.glob('./record_*.npy')
min_rmse = 99999
min_dim = 100
dim_rmse = {}
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
    dim_rmse[dim] = min(record['valid'])
print(min_dim, min_rmse)
dim_rmse = dict(sorted(dim_rmse.items()))
plt.plot(list(dim_rmse.keys()), list(dim_rmse.values()))
plt.xlabel("Dimension")
plt.ylabel("Minimum RMSE Achieved")
plt.title("Minimum validation RMSE achieved by each dimension")
plt.savefig('./plots/dim_rmse.png', bbox_inches='tight')
data = pd.read_csv(f'./test_pred_{min_dim}.csv')
for i in trange(len(data)):
    data.loc[i, 'stars'] = float(data.loc[i, 'stars'].replace('[', '').replace(']', ''))
data.to_csv('../data/pred.csv', index=False)
