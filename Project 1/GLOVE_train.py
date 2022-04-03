import argparse
import os
import zipfile

import numpy as np
import pandas as pd
import requests
import spacy
import torch
from sklearn.metrics import classification_report
from torch import nn
from torch.nn import BatchNorm1d
from torch.nn import CrossEntropyLoss
from torch.nn import Dropout
from torch.nn import LSTM
from torch.nn import Linear
from torch.optim import AdamW
from torch.utils.data import DataLoader
from torch.utils.data import Dataset
from tqdm import tqdm
from tqdm import trange
from transformers import get_scheduler


class TextDataset(Dataset):
    def __init__(self, mode):
        super(TextDataset, self).__init__()
        if mode not in ['train', 'valid', 'test']:
            raise NotImplementedError("mode must be in train, valid, test")
        self.mode = mode
        self.data = pd.read_csv(f'./data/{mode}.csv', index_col=None).drop(
            labels=[
                'business_id',
                'cool',
                'date',
                'funny',
                'review_id',
                'useful',
                'user_id',
            ],
            axis=1,
        )

    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        if self.mode in ['train', 'valid']:
            return self.data.loc[index, 'text'], self.data.loc[index, 'stars']-1
        else:
            return self.data.loc[index, 'text']


class BiLSTM_model(nn.Module):
    def __init__(self, spacy, encode_dict, glove_dim, sentence_length, bidirectional=True, hidden_size=256):
        super(BiLSTM_model, self).__init__()
        self.encode_dict = encode_dict
        self.glove_dim = glove_dim
        self.sentence_length = sentence_length
        self.bidirectional = bidirectional
        self.hidden_size = hidden_size
        self.nlp = spacy

        self.LSTM = LSTM(input_size=glove_dim, hidden_size=256, num_layers=2, bias=True, batch_first=True, dropout=0.2,
                         bidirectional=True)
        self.bn = BatchNorm1d(self.get_output_length())
        self.nn = Linear(in_features=self.get_output_length(), out_features=5)
        self.dropout = Dropout(p=0.2)

    def encode(self, text):
        doc = list(self.nlp(text))
        if len(doc) > self.sentence_length:
            doc = doc[:self.sentence_length]
        else:
            doc.extend(['-PAD-'] * (self.sentence_length - len(list(doc))))
        return torch.tensor([self.encode_dict[i] for i in doc])

    def forward(self, x):
        # print(x.shape)
        out, (h0, c0) = self.LSTM(x)
        # print(out.shape, h0.shape, c0.shape)
        seq_avg = torch.mean(out, dim=1).squeeze()  # (bs, 2 * hidden size)
        h0_avg = torch.mean(h0, dim=0).squeeze()  # (bs, hidden size)
        c0_avg = torch.mean(c0, dim=0).squeeze()  # (bs, hidden size)
        # print(seq_avg.shape, h0_avg.shape, c0_avg.shape)  # ,torch.cat((seq_avg, h0_avg, c0_avg), dim=1).shape)
        try:
            return self.dropout(self.nn(self.bn(torch.cat((seq_avg, h0_avg, c0_avg), dim=1))))
        except IndexError:
            return self.dropout(self.nn(self.bn(torch.cat((seq_avg, h0_avg, c0_avg), dim=-1).unsqueeze(0))))

    def get_output_length(self):
        return 4 * self.hidden_size if self.bidirectional else 3 * self.hidden_size


# Set up a argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, required=False, help="The number of gpu you want to use")
parser.add_argument("--epoch", type=int, default=100, required=False)
parser.add_argument("--bs", type=int, default=8, required=False)
parser.add_argument("--lr", type=float, default=1e-5, required=False)
parser.add_argument("--glove_size", type=int, choices=[6, 42, 840], default=840, required=False,
                    help="the number of how many billion words does the glove model pretrained on")
args = parser.parse_args()

glove_6B = 'http://downloads.cs.stanford.edu/nlp/data/glove.6B.zip'
glove_42B = 'http://downloads.cs.stanford.edu/nlp/data/glove.42B.300d.zip'
glove_840B = 'http://downloads.cs.stanford.edu/nlp/data/glove.840B.300d.zip'
if not os.path.exists('./glove'):
    os.makedirs('./glove')
if not os.path.isfile('./glove/glove.6B.zip'):
    print("\nDownloading glove.6B, please wait.\n")
    r_6b = requests.get(glove_6B, allow_redirects=True)
    open('./glove/glove.6B.zip', 'wb').write(r_6b.content)
    print("Downloading glove.6B Finished\n")
    with zipfile.ZipFile('./glove/glove.6B.zip', 'r') as zip_ref:
        zip_ref.extractall(path="./glove/")
    print("Glove 6B extracted")

if not os.path.isfile('./glove/glove.42B.300d.zip'):
    print("\nDownloading glove.42B, please wait.\n")
    r_42b = requests.get(glove_42B, allow_redirects=True)
    open('./glove/glove.42B.300d.zip', 'wb').write(r_42b.content)
    print("Downloading glove.42B Finished\n")
    with zipfile.ZipFile('./glove/glove.42B.300d.zip', 'r') as zip_ref:
        zip_ref.extractall(path="./glove/")
    print("Glove 42B extracted")

if not os.path.isfile('./glove/glove.840B.300d.zip'):
    print("\nDownloading glove.840B, please wait.\n")
    r_840b = requests.get(glove_840B, allow_redirects=True)
    open('./glove/glove.840B.300d.zip', 'wb').write(r_840b.content)
    print("Downloading glove.840B Finished\n")
    with zipfile.ZipFile('./glove/glove.840B.300d.zip', 'r') as zip_ref:
        zip_ref.extractall(path="./glove/")
    print("Glove 840B extracted")

# Set up some parameter we can use
glove_path_dict = {6: './glove/glove.6B.100d.txt', 42: './glove/glove.42B.300d.txt', 840: './glove/glove.840B.300d.txt'}
glove_dimension_dict = {6: 100, 42: 300, 840: 300}
epochs = args.epoch
BS = args.bs
LR = args.lr
glove_path = glove_path_dict[args.glove_size]
glove_dimension = glove_dimension_dict[args.glove_size]

# Read the glove embedding from the txt and save it into a dict
if not os.path.exists("./glove/encoded_dict_{}B_{}d.npy".format(args.glove_size, glove_dimension)):
    glove_dict = {}
    with open(glove_path, 'r', encoding="utf-8") as f:
        for line in tqdm(f, desc="loading GLOVE"):
            values = line.split(' ')
            word = values[0]
            vector = np.asarray(values[1:], "float32").reshape((1, -1))
            glove_dict[word] = vector

train_dataset = TextDataset('train')
valid_dataset = TextDataset('valid')
test_dataset = TextDataset('test')

# Give all the words appeared in our corpus their glove embedding, for those who are not exist, random initialize them
encoded_dict = {}
count, total = 0, 0
max_length = 1209
nlp = spacy.load('en_core_web_trf')

if os.path.exists("./glove/encoded_dict_{}B_{}d.npy".format(args.glove_size, glove_dimension)):
    encoded_dict = np.load("./glove/encoded_dict_{}B_{}d.npy".format(args.glove_size, glove_dimension),
                           allow_pickle=True).item()
else:
    glove_keys = glove_dict.keys()
    for i in [train_dataset, valid_dataset, test_dataset]:
        for j in trange(len(i)):
            text = i[j][0]
            if len(nlp(text)) > max_length:
                max_length = len(nlp(text))
            for token in nlp(text):
                if str(token) not in encoded_dict.keys():
                    if str(token) not in glove_keys:
                        encoded_dict[str(token)] = np.random.rand(1, glove_dimension)[0]
                        count += 1
                        total += 1
                    else:
                        encoded_dict[str(token)] = glove_dict[str(token)]
                        total += 1
    encoded_dict['-PAD-'] = np.zeros(shape=(1, glove_dimension), dtype=float)
    np.save("./glove/encoded_dict_{}B_{}d.npy".format(args.glove_size, glove_dimension), encoded_dict)

# Test how many words are found in glove and how many are randomly initialized
print("words not found {}".format(count))
print("words total {}".format(total))
print(len(encoded_dict))

# Load some parameters for deep learning
embedding_dim = glove_dimension
input_length = max_length
print(embedding_dim, input_length)

# Set the GPU device
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(args.gpu)
# Check device status
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA available:', torch.cuda.is_available())
print(torch.cuda.get_device_name())
print('Device number:', torch.cuda.device_count())
print(torch.cuda.get_device_properties(device))


def encode(text):
    doc = list(nlp(text))
    if len(doc) > max_length:
        doc = doc[:max_length]
    else:
        doc.extend(['-PAD-'] * (max_length - len(list(doc))))
    return torch.tensor([encoded_dict[str(i)].reshape(1, glove_dimension) for i in doc], dtype=torch.float)


model = BiLSTM_model(spacy=nlp, encode_dict=encoded_dict, glove_dim=embedding_dim, sentence_length=input_length).to(
    device)

train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BS, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=True)

criterion = CrossEntropyLoss()
optimizer = AdamW(model.parameters(), lr=LR)

lr_scheduler = get_scheduler(name='linear', optimizer=optimizer,
                             num_warmup_steps=0, num_training_steps=epochs * len(train_loader))

progress_bar = tqdm(range(epochs * len(train_loader)))

max_val_f1 = 0
max_f1_acc = 0
max_metrics = None

for e in trange(epochs, desc="Epoch: "):
    epoch_loss = 0

    model.train()
    total_acc, total_loss, total_count = 0, 0, 0
    for inputs, labels in tqdm(train_loader):
        inputs = torch.stack([encode(t) for t in inputs], dim=0).squeeze().to(device)
        labels = labels.type(torch.long).to(device)
        outputs = model(inputs)
        # print(outputs.shape,labels.shape)
        # print(outputs)
        # print(labels)
        loss = criterion(outputs, labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        predictions = torch.argmax(outputs, dim=-1)

        total_acc += (predictions == labels).sum().item()
        total_loss += loss.item()
        total_count += len(labels)

        progress_bar.update(1)
        progress_bar.set_postfix({'epoch': e,
                                  'loss': total_loss / total_count,
                                  'acc': total_acc / total_count})

    model.eval()
    y_valid_labels, y_pred_labels = [], []
    for inputs, labels in tqdm(valid_loader):
        inputs = torch.stack([encode(t) for t in inputs], dim=0).squeeze().to(device)
        labels = labels.type(torch.long).to(device)
        outputs = model(inputs)
        loss = criterion(outputs, labels)

        predictions = torch.argmax(outputs, dim=-1).cpu().numpy()
        y_valid_labels.append(labels.cpu().numpy())
        y_pred_labels.append(predictions)
    y_valid_labels = np.array(y_valid_labels).reshape(-1)
    y_pred_labels = np.array(y_pred_labels).reshape(-1)
    print(classification_report(y_valid_labels, y_pred_labels))
    macro_f1 = classification_report(y_valid_labels, y_pred_labels, output_dict=True)['macro avg']['f1-score']
    print('val accuracy', np.mean(y_valid_labels == y_pred_labels))
    if macro_f1 >= max_val_f1:
        max_val_f1 = macro_f1
        max_f1_acc = np.mean(y_valid_labels == y_pred_labels)
        max_metrics = classification_report(y_valid_labels, y_pred_labels)
        torch.save(model, './GLOVE_val_best.pkl')
    print('\n\n\n------------------------------------------\n'
          'MAX F1 {}\tMAX ACC {}\n{}'
          '-----------------------------------------------\n\n'.format(max_val_f1, max_f1_acc, max_metrics))
