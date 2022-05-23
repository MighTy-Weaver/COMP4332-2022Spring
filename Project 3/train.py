import argparse
import os
from math import sqrt

import numpy as np
import pandas as pd
import torch
from sklearn.metrics import mean_squared_error
from torch import nn
from torch.optim import AdamW
from torch.utils.data import DataLoader
from tqdm import tqdm
from tqdm import trange
from transformers import BertModel
from transformers import BertTokenizer
from transformers import get_scheduler

from model import NN_regressor

from dataloader import YelpDataset


def rmse(pred, actual):
    # Ignore nonzero terms.
    pred = pred[actual.nonzero()].flatten()
    actual = actual[actual.nonzero()].flatten()
    return sqrt(mean_squared_error(pred, actual))


# Set up a argument parser
parser = argparse.ArgumentParser()
parser.add_argument("--gpu", type=int, default=0, required=False, help="The number of gpu you want to use")
parser.add_argument("--epoch", type=int, default=300, required=False)
parser.add_argument("--bs", type=int, default=128, required=False)
parser.add_argument("--lr", type=float, default=5e-5, required=False)
parser.add_argument("--test", type=int, default=0)
args = parser.parse_args()

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

# Set up some parameter we can use
epochs = args.epoch
BS = args.bs
LR = args.lr

tokenizer = BertTokenizer.from_pretrained('bert-base-uncased')
encoder = BertModel.from_pretrained('bert-base-uncased').to(device)

train_dataset = YelpDataset('train', encoder=encoder, tokenizer=tokenizer, device=device, test=args.test)
valid_dataset = YelpDataset('valid', encoder=encoder, tokenizer=tokenizer, device=device, test=args.test)
test_dataset = YelpDataset('test', encoder=encoder, tokenizer=tokenizer, device=device, test=args.test)

model = NN_regressor(in_dim=6934, out_dim=1).to(device)

train_loader = DataLoader(train_dataset, batch_size=BS, shuffle=True)
valid_loader = DataLoader(valid_dataset, batch_size=BS, shuffle=True)
test_loader = DataLoader(test_dataset, batch_size=len(test_dataset), shuffle=False)

criterion = nn.MSELoss()
optimizer = AdamW(list(model.parameters()) + list(encoder.parameters()), lr=LR)

lr_scheduler = get_scheduler(name='linear', optimizer=optimizer,
                             num_warmup_steps=0, num_training_steps=epochs * len(train_loader))

min_RMSE = 999999

for e in trange(epochs, desc="Epoch: "):
    epoch_loss = 0

    model.train()
    train_label, train_pred = [], []
    for inputs, labels in tqdm(train_loader, desc="Train batch: "):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs).squeeze()
        # print(outputs.shape,labels.shape)
        # print(outputs)
        # print(labels)
        loss = torch.sqrt(criterion(outputs, labels))

        optimizer.zero_grad()
        loss.backward()
        optimizer.step()

        train_label.extend(labels.detach().cpu().tolist())
        train_pred.extend(outputs.detach().cpu().tolist())
    train_rmse = rmse(np.array(train_pred), np.array(train_label))
    model.eval()

    valid_pred, valid_label = [], []
    for inputs, labels in tqdm(valid_loader, desc="Valid batch"):
        inputs = inputs.to(device)
        labels = labels.to(device)
        outputs = model(inputs).squeeze()
        valid_loss = torch.sqrt(criterion(outputs, labels))

        valid_pred.extend(outputs.detach().cpu().tolist())
        valid_label.extend(labels.detach().cpu().tolist())
    valid_rmse = rmse(np.array(valid_pred), np.array(valid_label))
    if valid_rmse <= min_RMSE:
        min_RMSE = valid_rmse
        if not args.test:
            valid_csv = pd.read_csv('./data/valid.csv', index_col=None)
            valid_csv['stars_pred'] = valid_pred
            valid_csv.to_csv('./data/valid_pred.csv', index=False)

        y_test_labels = []
        for inputs in tqdm(test_loader, desc="Test batch: "):
            inputs = inputs.to(device)
            outputs = model(inputs)
            y_test_labels.extend(list(outputs.detach().cpu()))
        if not args.test:
            test_csv = pd.read_csv('./data/test.csv', index_col=None)
            test_csv['stars'] = y_test_labels
            test_csv.to_csv('./data/test_pred.csv', index=False)
    print('\n\n\n------------------------------------------\n'
          'MIN RMSE valid {}\ttrain {} valid {}\n'
          '-----------------------------------------------\n\n'.format(min_RMSE, train_rmse, valid_rmse))
