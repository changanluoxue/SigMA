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
class sigformer(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(sigformer, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        # (time_grid, in_features): (100,310), (200,620), (300,930)
        self.dense = fnn(input_size=930, hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
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
        x = torch.stack([layers.Sig(x[:, :50*i, :], 3) for i in range(1, 1+int(x.size(1)/50))])
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
        self.attention = layers.SingleAttention(d_k=155, in_features=14)
        self.dense = nn.Linear(in_features=155, out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = layers.Sig(x, 3)
        x = self.attention(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

'sigformer_1 is sigformer without convolutional layer'
class sigformer_1(nn.Module):
    def __init__(self, augment_include_time=True, T=1):
        super(sigformer_1, self).__init__()
        self.augment_include_time = augment_include_time
        self.T = T
        self.attention = layers.MultiHeadAttention(d_k=[2, 4, 8], in_features=[2, 4, 8], heads=3, out_features=14)
        # (time_grid, in_features): (100,28), (200,56), (300,84)
        self.dense = fnn(input_size=84, hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :50*i, :], 3) for i in range(1, 1+int(x.size(1)/50))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

'sigformer_2 is sigformer without MLP'
class sigformer_2(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(sigformer_2, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        # (time_grid, in_features): (100,310), (200,620), (300,930)
        self.dense = nn.Linear(in_features=930, out_features=1)
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
        x = torch.stack([layers.Sig(x[:, :50*i, :], 3) for i in range(1, 1+int(x.size(1)/50))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

'sigformer_3 is sigformer without convolutionsl layer and MLP'
class sigformer_3(nn.Module):
    def __init__(self, augment_include_time=True, T=1):
        super(sigformer_3, self).__init__()
        self.augment_include_time = augment_include_time
        self.T = T
        self.attention = layers.MultiHeadAttention(d_k=[2, 4, 8], in_features=[2, 4, 8], heads=3, out_features=14)
        # (time_grid, in_features): (100,28), (200,56), (300,84)
        self.dense = nn.Linear(in_features=84, out_features=1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :50*i, :], 3) for i in range(1, 1+int(x.size(1)/50))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x