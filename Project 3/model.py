import torch
from torch import nn


class NN_regressor(nn.Module):
    def __init__(self, in_dim=6935, out_dim=1):
        super(NN_regressor, self).__init__()
        self.nn1 = nn.Sequential(nn.Linear(in_features=in_dim, out_features=4096), nn.BatchNorm1d(num_features=4096),
                                 nn.Dropout(0.2))
        self.nn2 = nn.Sequential(nn.Linear(in_features=4096, out_features=1024), nn.BatchNorm1d(num_features=1024),
                                 nn.Dropout(0.2))
        self.nn3 = nn.Sequential(nn.Linear(in_features=1024, out_features=256), nn.BatchNorm1d(num_features=256),
                                 nn.Dropout(0.2))
        self.nn4 = nn.Sequential(nn.Linear(in_features=256, out_features=64), nn.BatchNorm1d(num_features=64),
                                 nn.Dropout(0.2))
        self.nn5 = nn.Linear(in_features=64, out_features=out_dim)

    def forward(self, x):
        out1 = self.nn1(x)
        out2 = self.nn2(out1)
        out3 = self.nn3(out2)
        out4 = self.nn4(out3)
        return self.nn5(out4)


class NN_regressor_v2(nn.Module):
    def __init__(self, in_dim_1=6912, in_dim_2=22, out_dim=1):
        super(NN_regressor_v2, self).__init__()
        self.flatten = nn.Flatten()
        self.nn1 = nn.Sequential(nn.Linear(in_features=in_dim_1, out_features=2048), nn.BatchNorm1d(num_features=2048),
                                 nn.Dropout(0.2))
        self.nn2 = nn.Sequential(nn.Linear(in_features=2048, out_features=512), nn.BatchNorm1d(num_features=512),
                                 nn.Dropout(0.2))
        self.nn3 = nn.Sequential(nn.Linear(in_features=512, out_features=128), nn.BatchNorm1d(num_features=128),
                                 nn.Dropout(0.2))
        self.nn4 = nn.Sequential(nn.Linear(in_features=128, out_features=32), nn.BatchNorm1d(num_features=32),
                                 nn.Dropout(0.2))
        self.feature_nn1 = nn.Sequential(nn.Linear(in_features=in_dim_2, out_features=32),
                                         nn.BatchNorm1d(num_features=32), nn.Dropout(0.2))
        self.final_nn1 = nn.Sequential(nn.Linear(in_features=64, out_features=16), nn.BatchNorm1d(num_features=16),
                                       nn.Dropout(0.2))
        self.final_nn2 = nn.Linear(in_features=16, out_features=out_dim)

    def forward(self, embedding_x, feature_x):
        out1 = self.flatten(embedding_x)
        out1 = self.nn1(out1)
        out1 = self.nn2(out1)
        out1 = self.nn3(out1)
        out1 = self.nn4(out1)
        out2 = self.feature_nn1(feature_x)
        out = torch.cat([out1, out2], dim=-1)
        out3 = self.final_nn1(out)
        return self.final_nn2(out3)
