import torch
from torch import nn


class NN_regressor(nn.Module):
    def __init__(self, user_dict, business_dict, embedding_dim=30, out_dim=1):
        super(NN_regressor, self).__init__()
        self.user_dict = user_dict
        self.business_dict = business_dict
        self.embedding_dim = embedding_dim
        self.user_embedding = nn.Embedding(num_embeddings=len(self.user_dict) + 1, embedding_dim=embedding_dim)
        self.business_embedding = nn.Embedding(num_embeddings=len(self.business_dict) + 1, embedding_dim=embedding_dim)
        self.user_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=128, num_layers=2, bias=True, bidirectional=True,
                                 batch_first=True, dropout=0.2)
        self.business_LSTM = nn.LSTM(input_size=embedding_dim, hidden_size=128, num_layers=2, bias=True,
                                     bidirectional=True, batch_first=True, dropout=0.2)
        self.user_nn1 = nn.Sequential(nn.Linear(in_features=256, out_features=128), nn.BatchNorm1d(num_features=128),
                                      nn.Dropout(0.2))
        self.user_nn2 = nn.Sequential(nn.Linear(in_features=128, out_features=64), nn.BatchNorm1d(num_features=64),
                                      nn.Dropout(0.2))
        self.business_nn1 = nn.Sequential(nn.Linear(in_features=256, out_features=128),
                                          nn.BatchNorm1d(num_features=128),
                                          nn.Dropout(0.2))
        self.business_nn2 = nn.Sequential(nn.Linear(in_features=128, out_features=64), nn.BatchNorm1d(num_features=64),
                                          nn.Dropout(0.2))
        self.feature_nn1 = nn.Sequential(nn.Linear(in_features=128, out_features=64), nn.BatchNorm1d(num_features=64),
                                         nn.Dropout(0.2))
        self.final_nn1 = nn.Sequential(nn.Linear(in_features=64, out_features=16), nn.BatchNorm1d(num_features=16),
                                       nn.Dropout(0.2))
        self.final_nn2 = nn.Linear(in_features=16, out_features=out_dim)
        self.relu = nn.ReLU()

    def forward(self, user_id, business_id):
        user_embedding = self.user_embedding(user_id)
        business_embedding = self.business_embedding(business_id)
        user_lstm, _ = self.user_LSTM(user_embedding)
        business_lstm, _ = self.business_LSTM(business_embedding)
        user_lstm = torch.mean(user_lstm, dim=1).squeeze()
        business_lstm = torch.mean(business_lstm, dim=1).squeeze()
        user_nn = self.user_nn2(self.user_nn1(user_lstm))
        business_nn = self.business_nn2(self.business_nn1(business_lstm))
        total_nn = torch.cat([user_nn, business_nn], dim=-1)
        output = self.relu(self.final_nn1(self.feature_nn1(total_nn)))
        return self.final_nn2(output)
