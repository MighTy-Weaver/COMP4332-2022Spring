import pandas as pd
import torch
from torch.utils.data import Dataset
from tqdm import tqdm


class YelpDataset(Dataset):
    def __init__(self, mode='train', test=0, drop_text=True):
        if mode not in ['train', 'valid', 'test']:
            raise NotImplementedError("Must be one of train, valid, and test")
        self.mode = mode
        self.data = pd.read_csv(f'../data/{mode}.csv', index_col=None)
        self.total_data = pd.concat(
            [pd.read_csv('../data/train.csv', index_col=None), pd.read_csv('../data/valid.csv', index_col=None),
             pd.read_csv('../data/test.csv', index_col=None)], ignore_index=True)
        user_df = pd.read_csv("../data/user.csv", index_col=0)
        item_df = pd.read_csv("../data/business.csv", index_col=0)
        user_df = user_df.rename(index=str, columns={t: 'user_' + t for t in user_df.columns if t != 'user_id'})
        item_df = item_df.rename(index=str, columns={t: 'business_' + t for t in item_df.columns if t != 'business_id'})
        self.data_merged = pd.merge(pd.merge(self.data, user_df, on='user_id'), item_df, on='business_id').reset_index(
            drop=True).drop(['user_yelping_since', 'user_elite', 'business_attributes', 'business_hours'],
                            axis=1).reset_index(drop=True)
        self.total_data_merged = pd.merge(pd.merge(self.total_data, user_df, on='user_id'), item_df,
                                          on='business_id').reset_index(drop=True).drop(
            ['user_yelping_since', 'user_elite', 'business_attributes', 'business_hours'], axis=1)
        self.str_list = ['user_id', 'business_id', 'user_name', 'business_name', 'business_address', 'business_city',
                         'business_state', 'business_postal_code', 'business_categories']
        if drop_text:
            self.total_data_merged = self.total_data_merged.drop(self.str_list, axis=1)
            self.data_merged = self.data_merged.drop(self.str_list, axis=1)
        self.user_columns = [i for i in list(self.data_merged) if 'user_' in i]
        self.business_columns = [i for i in list(self.data_merged) if 'business_' in i]

        self.user_encoded_idx_dict = {}
        self.business_encoded_idx_dict = {}
        for c in tqdm(list(self.total_data_merged)):
            if c in self.user_columns:
                for u in self.total_data_merged[c].unique():
                    self.user_encoded_idx_dict[u] = len(self.user_encoded_idx_dict)
            elif c in self.business_columns:
                for u in self.total_data_merged[c].unique():
                    self.business_encoded_idx_dict[u] = len(self.business_encoded_idx_dict)

        if test:
            self.data_merged = self.data_merged.sample(n=200).reset_index(drop=True)
        self.user_merged = self.data_merged[self.user_columns].replace(to_replace=self.user_encoded_idx_dict)
        self.business_merged = self.data_merged[self.business_columns].replace(
            to_replace=self.business_encoded_idx_dict)

    def __len__(self):
        return len(self.data_merged)

    def __getitem__(self, item):
        user = self.user_merged.loc[item].to_numpy()
        business_df = self.business_merged.loc[item].to_numpy()
        if self.mode in ['train', 'valid']:
            return torch.tensor(user, dtype=torch.long), torch.tensor(business_df, dtype=torch.long), torch.tensor(
                self.data_merged.loc[item, 'stars'], dtype=torch.float)
        else:
            return torch.tensor(user, dtype=torch.long), torch.tensor(business_df, dtype=torch.long)

    def get_user_dict(self):
        return self.user_encoded_idx_dict

    def get_business_dict(self):
        return self.business_encoded_idx_dict
