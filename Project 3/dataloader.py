import pandas as pd
import torch
from torch.utils.data import Dataset
from transformers import BertTokenizer, BertModel


class YelpDataset(Dataset):
    def __init__(self, mode='train', encoder=None, tokenizer=None, device=None):
        if mode not in ['train', 'valid', 'test']:
            raise NotImplementedError("Must be one of train, valid, and test")
        self.mode = mode
        self.data = pd.read_csv(f'./data/{mode}.csv', index_col=None)
        self.device = device
        self.encoder = encoder
        self.tokenizer = tokenizer
        user_df = pd.read_csv("data/user.csv", index_col=0)
        item_df = pd.read_csv("data/business.csv", index_col=0)
        user_df = user_df.rename(index=str, columns={t: 'user_' + t for t in user_df.columns if t != 'user_id'})
        item_df = item_df.rename(index=str, columns={t: 'item_' + t for t in item_df.columns if t != 'business_id'})
        self.data_merged = pd.merge(pd.merge(self.data, user_df, on='user_id'), item_df, on='business_id').reset_index(
            drop=True).drop(['user_yelping_since', 'user_elite', 'item_attributes', 'item_hours'], axis=1)
        if self.mode in ['train', 'valid']:
            self.stars = self.data_merged['stars']
            self.data_merged = self.data_merged.drop(['stars'], axis=1)

    def __len__(self):
        return len(self.data_merged)

    def __getitem__(self, item):
        data_list = []
        for c in list(self.data_merged):
            if type(self.data_merged.loc[item, c]) is str:
                token = self.tokenizer(self.data_merged.loc[item, c], return_tensors="pt").to(self.device)
                embedding = self.encoder(**token)
                data_list.extend(torch.mean(embedding.last_hidden_state, dim=1).detach().cpu().squeeze().tolist())
            else:
                data_list.append(self.data_merged.loc[item, c])
        embedding_final = torch.tensor(data_list, dtype=torch.float)
        if self.mode in ['train', 'valid']:
            return embedding_final, torch.tensor(self.data.loc[item, 'stars'], dtype=torch.float)
        else:
            return embedding_final
