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
    def __init__(self, in_dim_1, in_dim_2, embedding_dim=798, out_dim=1):
        super(NN_regressor_v2, self).__init__()

