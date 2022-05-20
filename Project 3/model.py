from torch import nn


class NN_regressor(nn.Module):
    def __init__(self, in_dim=6935, out_dim=1):
        self.nn1=nn.Sequential()