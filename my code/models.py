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
        self.dense1 = nn.Linear(in_features=384, out_features=128)# (time_grid, in_features): (100,384), (300,1408), (500,2304)
        self.dense2 = nn.Linear(in_features=128, out_features=1)
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
        x = self.sigmoid(x)
        return x
class cnn_simple(nn.Module):
    def __init__(self):
        '''
        Arg:
            - Input shape: (batch, channel, seq)
        Examples:
            input = torch.rand((64, 1, 500))
            model = cnn_simple()
            output = model(input)
        '''
        super(cnn_simple, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        self.dense1 = nn.Linear(in_features=3200, out_features=128)# (time_grid, in_features): (300,3200), (500,5312)
        self.dense2 = nn.Linear(in_features=128, out_features=1)
        self.LRelu = nn.LeakyReLU(negative_slope=0.1)
        self.pooling = nn.MaxPool1d(kernel_size=3)
        self.dropout = nn.Dropout1d(p=0.25)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.LRelu(x)
        x = self.pooling(x)
        x = self.dropout(x)

        x = self.flatten(x)
        x = self.dense1(x)
        x = self.LRelu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x
class cnn_sig(nn.Module):
    def __init__(self):
        '''
        Arg:
            - Input shape: (batch, channel, seq)
        Examples:
            input = torch.rand((64, 1, 100))
            model = cnn_sig()
            output = model(input)
        '''
        super(cnn_sig, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        self.dense1 = nn.Linear(in_features=1056, out_features=128)
        self.dense2= nn.Linear(in_features=128, out_features=1)
        self.LRelu = nn.LeakyReLU(negative_slope=0.1)
        self.dropout = nn.Dropout1d(p=0.25)
        self.sigmoid = nn.Sigmoid()
    def forward(self, x):
        x = self.conv1(x)
        x = self.LRelu(x)
        x = torch.transpose(x, 1, 2)
        x = layers.Sig(x, 2)
        x = self.dropout(x)

        x = self.dense1(x)
        x = self.LRelu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x
class my_deepsignet(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(my_deepsignet, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.dense = fnn(input_size=155, hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
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
        x = self.sigmoid(x)
        return x
class deepsignet_transformer_1(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(deepsignet_transformer_1, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.SingleAttention(d_k=155, in_features=155)
        self.dense = fnn(input_size=15345, hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())# (time_grid, in_features): (100,15345)
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
        x = torch.stack([layers.Sig(x[:, :i, :], 3) for i in range(2, 1 + x.size(1))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x
# class multi_head_attention(nn.Module):
#     def __init__(self, d_model=128, nhead=5, num_layers=5):
#         super(multi_head_attention, self).__init__()
#         self.encoder = nn.TransformerEncoder(nn.TransformerEncoderLayer(d_model=d_model, nhead=nhead, batch_first=True), num_layers=num_layers)
#     def forward(self, x):
#         x = self.encoder(x)
#         return x
# class deepsignet_transformer_2(nn.Module):
#     def __init__(self, d_model=155, nhead=5, attention_layers=1, augment_include_original=True, augment_include_time=True, T=1):
#         super(deepsignet_transformer_2, self).__init__()
#         self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
#         self.attention = multi_head_attention(d_model=d_model, nhead=nhead, num_layers=attention_layers)
#         self.dense = nn.Linear(in_features=155, out_features=1)
#         self.sigmoid = nn.Sigmoid()
#         self.augment_include_original = augment_include_original
#         self.augment_include_time = augment_include_time
#         self.T = T
#     def forward(self, x):
#         if self.augment_include_original is True:
#             value = x
#         if self.augment_include_time is True:
#             time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, x.shape[-1])
#             time = time.expand(x.shape[0], 1, x.shape[-1])
#         x = self.conv(x)
#         if self.augment_include_original is True:
#             x = torch.cat((x, value), dim=1)
#         if self.augment_include_time is True:
#             x = torch.cat((x, time), dim=1)
#         x = torch.transpose(x, 1, 2)
#         x = layers.Sig(x, 3)
#         x = self.attention(x)
#         x = self.dense(x)
#         x = self.sigmoid(x)
#         return x
class deepsignet_transformer_2(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(deepsignet_transformer_2, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.SingleAttention(d_k=155, in_features=155)
        self.dense = nn.Linear(in_features=15345, out_features=1)
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
        x = torch.stack([layers.Sig(x[:, :i, :], 3) for i in range(2, 1 + x.size(1))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

class deepsignet_transformer_3(nn.Module):
    def __init__(self, augment_include_time=True, T=1):
        super(deepsignet_transformer_3, self).__init__()
        self.augment_include_time = augment_include_time
        self.T = T
        self.attention = layers.SingleAttention(d_k=5, in_features=2)
        self.dense = fnn(input_size=15345, hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
    def forward(self, x):
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1]).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = self.attention(x)
        x = torch.stack([layers.Sig(x[:, :i, :], 3) for i in range(2, 1 + x.size(1))])
        x = torch.transpose(x, 0, 1)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

# import utils
# input = torch.rand(64,1,100)
# model = deepsignet_transformer_3()
# print(model(input).shape)
# print(utils.count_parameters(model))