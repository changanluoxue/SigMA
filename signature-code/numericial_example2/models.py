import torch
import torch.nn as nn
import layers

def fnn(input_size=602, hidden_sizes=(16, 16, 16), output_size=1, activation=nn.ReLU()):
    layers = []
    in_size = input_size
    for size in hidden_sizes:
        layers.append(nn.Linear(in_size, size))
        layers.append(activation)
        in_size = size
    layers.append(nn.Linear(in_size, output_size))
    model = nn.Sequential(*layers)
    return model
class deepsignet(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(deepsignet, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        # truncation order (1,5),(2,30),(3,155),(4,780),(5,3905)
        self.dense = fnn(input_size=3905, hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
        x = self.conv(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        # truncation order 1,2,3,4,5
        x = layers.Sig(x, 5)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x
class sigformer(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(sigformer, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        # truncation order (1,5),(2,30),(3,155),(4,780),(5,3905); [5,25,125,625,3125]
        self.attention = layers.MultiHeadAttention(d_k=[5,25,125,625,3125], in_features=[5,25,125,625,3125], heads=5, out_features=3905)
        # set time_grid=100, then (truncation order, input_size): (1,10),(2,60),(3,310),(4,1560),(5,7810)
        self.dense = fnn(input_size=7810, hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
        x = self.conv(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        # truncation order 1,2,3,4,5
        x = torch.stack([layers.Sig(x[:, :50*i, :], 5) for i in range(1, 1+int(x.size(1)/50))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x
class sigformer_s(nn.Module):
    def __init__(self, augment_include_time=True, T=1):
        super(sigformer_s, self).__init__()
        self.augment_include_time = augment_include_time
        self.T = T
        # (truncation order, in_features): (1,2),(2,6),(3,14),(4,30),(5,62)
        self.attention = layers.SingleAttention(d_k=155, in_features=62)
        self.dense = nn.Linear(in_features=155, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        # truncation order 1,2,3,4,5
        x = layers.Sig(x, 5)
        x = self.attention(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x




