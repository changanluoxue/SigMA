import torch
import torch.nn as nn
import utils

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
        self.dense1 = nn.Linear(in_features=2304, out_features=128)
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
        self.dense1 = nn.Linear(in_features=5312, out_features=128)
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
        x = utils.Sig(x, 2)
        x = self.dropout(x)

        x = self.dense1(x)
        x = self.LRelu(x)
        x = self.dropout(x)
        x = self.dense2(x)
        x = self.sigmoid(x)
        return x
class Transformer(nn.Module):
    def __init__(self, d_model=512, nhead=8, num_encoder_layers=6, num_decoder_layers=6, dim_feedforward=2048, dropout=0.1,
                 layer_norm_eps=1e-5, batch_first=False, norm_first=False, **kwargs):
        '''
        Args:
            - d_model (int): the number of expected features in the encoder/decoder inputs (default=512).
            - nhead (int): the number of heads in the multiheadattention models (default=8).
            - num_encoder_layers (int): the number of sub-encoder-layers in the encoder (default=6).
            - num_decoder_layers (int): the number of sub-decoder-layers in the decoder (default=6).
            - dim_feedforward (int): the dimension of the feedforward network model (default=2048).
            - dropout (float): the dropout value (default=0.1).
            - layer_norm_eps (float): the eps value in layer normalization components (default=1e-5).
            - batch_first (bool): If True, then the input and output tensors are provided as (batch, seq, feature). Default: False (seq, batch, feature).
            - norm_first (bool): if True, encoder and decoder layers will perform LayerNorms before other attention and feedforward operations, otherwise after. Default: False (after).
        Examples:
            transformer_model = nn.Transformer(nhead=16, num_encoder_layers=12)
            src = torch.rand((10, 32, 512))
            tgt = torch.rand((20, 32, 512))
            out = transformer_model(src, tgt)
        '''
        super(Transformer, self).__init__(**kwargs)
        self.mod = nn.Transformer(d_model=d_model, nhead=nhead, num_encoder_layers=num_encoder_layers, num_decoder_layers=num_decoder_layers,
                                  dim_feedforward=dim_feedforward, dropout=dropout, layer_norm_eps=layer_norm_eps, batch_first=batch_first, norm_first=norm_first)
    def forward(self, x, y):
        return self.mod(x, y)
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
        x = utils.Sig(x, 3)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

# import utils
# input = torch.rand(64,1,500)
# model = cnn_simple()
# print(model(input).shape)
