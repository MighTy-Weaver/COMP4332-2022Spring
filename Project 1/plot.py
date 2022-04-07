import matplotlib.pyplot as plt

import numpy as np

record = np.load('./BERT_recording.npy', allow_pickle=True).item()
print(record)
text_dict = {'loss': 'Loss', 'acc': 'Accuracy'}
for part in ['loss', 'acc']:
    x = range(1, len(record[f'trn_{part}']) + 1)
    plt1 = plt.plot(x, record[f'trn_{part}'], color='red', label='training')
    plt2 = plt.plot(x, record[f'val_{part}'], color='blue', label='validation')
    plt.title(f"{text_dict[part]} curve for BERT-base training process")
    plt.xlabel("Number of epochs")
    plt.legend(loc=7, frameon=False)
    plt.ylabel(f"{text_dict[part]}")
    plt.savefig(f'./{part}.png', bbox_inches='tight')
    plt.clf()
