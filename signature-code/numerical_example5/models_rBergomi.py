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
class cnn(nn.Module):
    def __init__(self):
        '''
        This CNN architecture is from https://doi.org/10.1080/14697688.2019.1654126
        Arg:
            - Input shape: (batch, channel, seq)
        Examples:
            input = torch.rand((64, 1, 500))
            model = cnn()
            output = model(input)
        '''
        super(cnn, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        # (time_grid, in_features): (500,2304)
        self.dense1 = nn.Linear(in_features=2304, out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=2)
        self.LRelu1 = nn.LeakyReLU(negative_slope=0.1)
        self.LRelu2 = nn.LeakyReLU(negative_slope=0.3)
        self.pooling = nn.MaxPool1d(kernel_size=3)
        self.dropout1 = nn.Dropout1d(p=0.25)
        self.dropout2 = nn.Dropout1d(p=0.4)
        self.dropout3 = nn.Dropout1d(p=0.3)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.LRelu1(x)
        x = self.pooling(x)
        x = self.dropout1(x)

        x = self.conv2(x)
        x = self.LRelu2(x)
        x = self.pooling(x)
        x = self.dropout1(x)

        x = self.conv3(x)
        x = self.LRelu1(x)
        x = self.pooling(x)
        x = self.dropout2(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.LRelu1(x)
        x = self.dropout3(x)
        x = self.dense2(x)
        return x
class deepsignet(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(deepsignet, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.dense = fnn(input_size=155, hidden_sizes=(32, 32, 32, 32, 32), output_size=2, activation=nn.ReLU())
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
        x = layers.Sig(x, 3)
        x = self.dense(x)
        return x
class sigformer(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(sigformer, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        # (time_grid, in_features): (500,310)
        self.dense = fnn(input_size=310, hidden_sizes=(32, 32, 32, 32, 32), output_size=2, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_layer = nn.Linear(2, 2, bias=False)
        self.linear_layer.weight.data = torch.tensor([[1,0], [0,3]], dtype=torch.float32)
        self.linear_layer.weight.requires_grad = False

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
        x = torch.stack([layers.Sig(x[:, :250*i, :], 3) for i in range(1, 1+int(x.size(1)/250))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        x = self.linear_layer(x)
        return x
class transformer(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(transformer, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=153, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        # (time_grid, in_features): (500,77500)
        self.dense = fnn(input_size=77500, hidden_sizes=(32, 32, 32, 32, 32), output_size=2, activation=nn.ReLU())
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
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

