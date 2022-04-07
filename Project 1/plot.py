import matplotlib.pyplot as plt

import numpy as np

record = np.load('./BERT_recording.npy', allow_pickle=True).item()
print(record)
text_dict = {'loss': 'Loss', 'acc': 'Accuracy'}
for part in ['loss', 'acc']:
    x = range(1, len(record[f'trn_{part}']) + 1)
    plt1 = plt.plot(x, record[f'trn_{part}'], color='red', label='training')
    plt2 = plt.plot(x, record[f'val_{part}'], color='blue', label='validation')
    plt.title("{} curve for BERT-base training process".format(text_dict[part]))
    plt.xlabel("Number of epochs")
    plt.legend(loc=7)
    plt.ylabel("{}".format(text_dict[part]))
    plt.savefig('./{}.png'.format(part), bbox_inches='tight')
    plt.clf()
