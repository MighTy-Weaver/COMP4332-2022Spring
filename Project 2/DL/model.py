from torch import nn


class NN(nn.Module):
    def __init__(self):
        super(NN, self).__init__()

        self.nn1 = nn.Sequential(nn.Linear(in_features=20, out_features=256), nn.BatchNorm1d(num_features=256),
                                 nn.Dropout(0.25))
        self.nn2 = nn.Sequential(nn.Linear(in_features=256, out_features=128), nn.BatchNorm1d(num_features=128),
                                 nn.Dropout(0.25))
        self.nn3 = nn.Sequential(nn.Linear(in_features=128, out_features=64), nn.BatchNorm1d(num_features=64),
                                 nn.Dropout(0.25))
        self.nn4 = nn.Linear(in_features=64, out_features=32)
        self.nn5 = nn.Linear(in_features=32, out_features=1)

    def forward(self, x):
        out = self.nn1(x)
        out = self.nn2(out)
        out = self.nn3(out)
        out = self.nn4(out)
        out = self.nn5(out)
        return out
