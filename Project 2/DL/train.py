import argparse
import os

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import classification_report
from sklearn.metrics import roc_auc_score
from torch.nn import MSELoss

from model import NN
from torch.nn import CrossEntropyLoss
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange
from transformers import get_scheduler
from dataloader import SocialNetworkDataset

# Set up a argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, required=False, help="The number of gpu you want to use")
parser.add_argument("--epoch", type=int, default=500, required=False)
parser.add_argument("--bs", type=int, default=128, required=False)
parser.add_argument("--lr", type=float, default=5e-5, required=False)
parser.add_argument("--test", type=int, default=0)
args = parser.parse_args()

embedding = np.load('./embedding.npy', allow_pickle=True).item()

train_dataset = SocialNetworkDataset(embedding=embedding, mode='train', negative_sample='same', test=args.test)
valid_dataset = SocialNetworkDataset(embedding=embedding, mode='valid', total_length=40000, test=args.test)

# Check device status
device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
print('CUDA available:', torch.cuda.is_available())
print(torch.cuda.get_device_name())
print('Device number:', torch.cuda.device_count())
print(torch.cuda.get_device_properties(device))

# Set the GPU device
if torch.cuda.is_available():
    os.environ["CUDA_VISIBLE_DEVICES"] = str(args.gpu)
    torch.cuda.set_device(args.gpu)

model = NN().to(device)

train_loader = DataLoader(train_dataset, batch_size=args.bs, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=args.bs, shuffle=True)
# test_loader = DataLoader(test_dataset, batch_size=BS, shuffle=False)

criterion = MSELoss()
optimizer = AdamW(model.parameters(), lr=args.lr)

lr_scheduler = get_scheduler(name='linear', optimizer=optimizer,
                             num_warmup_steps=0, num_training_steps=args.epoch * len(train_loader))

val_auc, trn_auc = [], []

progress_bar = tqdm(range(args.epoch * len(train_loader)))

for e in range(args.epoch):
    epoch_loss = 0

    model.train()
    total_loss, total_count = 0, 0
    total_label, total_pred = [], []
    for inputs, labels in train_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1), labels)

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        total_label.extend(list(labels.detach().cpu().numpy()))
        total_pred.extend(list(outputs.detach().cpu().reshape(-1)))
        total_loss += loss.item()
        total_count += len(labels)

        progress_bar.update(1)
        progress_bar.set_postfix({'epoch': e,
                                  'loss': total_loss / total_count,
                                  'auc': roc_auc_score(total_label, total_pred)})
    trn_auc.append(roc_auc_score(total_label, total_pred))

    model.eval()
    y_valid_labels, y_pred_labels = [], []
    for inputs, labels in valid_loader:
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs)
        loss = criterion(outputs.reshape(-1), labels)

        y_valid_labels.extend(list(labels.detach().cpu().numpy()))
        y_pred_labels.extend(list(outputs.detach().cpu().reshape(-1)))
    y_valid_labels = np.array(y_valid_labels).reshape(-1)
    y_pred_labels = np.array(y_pred_labels).reshape(-1)
    print('val auc', roc_auc_score(y_valid_labels, y_pred_labels))
    val_auc.append(roc_auc_score(y_valid_labels, y_pred_labels))
    print('MAX VAL AUC IS {}'.format(max(val_auc)))
    if roc_auc_score(y_valid_labels, y_pred_labels) >= max(val_auc):
        test = pd.read_csv('../data/test.csv', index_col=None)
        input_list = []
        for i in range(len(test)):
            if test.loc[i, 'src'] in embedding.keys():
                src_embedding = embedding[test.loc[i, 'src']]
            else:
                src_embedding = np.random.rand(10)
            if test.loc[i, 'dst'] in embedding.keys():
                dst_embedding = embedding[test.loc[i, 'dst']]
            else:
                dst_embedding = np.random.rand(10)
            input_list.append(np.array(list(src_embedding) + list(dst_embedding)))
        inputs = torch.tensor(input_list, dtype=torch.float32).to(device)
        outputs = model(inputs)
        score = outputs.detach().cpu().numpy()
        test['score'] = score
        test.to_csv('../data/pred_nn.csv', index=False)
