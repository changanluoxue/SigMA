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
class Standardization(nn.Module):
    def __init__(self):
        super(Standardization, self).__init__()
    def forward(self, x):
        x_diff = x[:,:,1:]-x[:,:,:-1]
        mean = x_diff.mean(dim=-1, keepdim=True)
        std = x_diff.std(dim=-1, keepdim=True)
        std = std + 1e-6
        x_standardized = (x_diff-mean)/std
        return x_standardized

'''Example1: Stride Sensitivity Analysis'''
class sigma_stride(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1, grid_points=100, stride=1):
        super(sigma_stride, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.stride = stride
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        self.dense = fnn(input_size=int(grid_points/stride*155), hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
        x = self.conv(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :self.stride*i, :], 3) for i in range(1, 1+int(x.size(1)/self.stride))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

'''Example2: Truncation Order Sensitivity Analysis'''
class deepsignet_truncation(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1, truncation_order=None):
        super(deepsignet_truncation, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.dense = fnn(input_size=int(sum([5,25,125,625,3125][0:truncation_order])), hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.truncation_order = truncation_order
    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
        x = self.conv(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = layers.Sig(x, self.truncation_order)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x
class sigma_truncation(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1, stride=None, truncation_order=None):
        super(sigma_truncation, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.stride = stride
        self.truncation_order = truncation_order
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5,25,125,625,3125][0:truncation_order], in_features=[5,25,125,625,3125][0:truncation_order],
                                                   heads=truncation_order, out_features=int(sum([5,25,125,625,3125][0:truncation_order])))
        self.dense = fnn(input_size=int(2*sum([5,25,125,625,3125][0:truncation_order])), hidden_sizes=(32, 32, 32, 32, 32),
                         output_size=1, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
        x = self.conv(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :self.stride*i, :], self.truncation_order) for i in range(1, 1+int(x.size(1)/self.stride))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x
class sigsa_truncation(nn.Module):
    def __init__(self, augment_include_time=True, T=1, stride=None, truncation_order=None):
        super(sigsa_truncation, self).__init__()
        self.augment_include_time = augment_include_time
        self.T = T
        self.stride = stride
        self.truncation_order = truncation_order
        self.attention = layers.SingleAttention(d_k=int(2**(truncation_order+1)-2), in_features=int(2**(truncation_order+1)-2))
        self.dense = nn.Linear(in_features=int(2**(truncation_order+1)-2), out_features=1)
        self.sigmoid = nn.Sigmoid()

    def forward(self, x):
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = layers.Sig(x, self.truncation_order)
        x = self.attention(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x

'''Example3: Architecture of MLP and CNN'''
class sigma_architecture0(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1, stride=None):
        super(sigma_architecture0, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.stride = stride
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        self.dense = fnn(input_size=310, hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
        x = self.conv(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :self.stride*i, :], 3) for i in range(1, 1+int(x.size(1)/self.stride))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x
'sigma_architecture1 is sigma without convolutional layer'
class sigma_architecture1(nn.Module):
    def __init__(self, augment_include_time=True, T=1, stride=None):
        super(sigma_architecture1, self).__init__()
        self.augment_include_time = augment_include_time
        self.T = T
        self.stride = stride
        self.attention = layers.MultiHeadAttention(d_k=[2, 4, 8], in_features=[2, 4, 8], heads=3, out_features=14)
        self.dense = fnn(input_size=28, hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :self.stride*i, :], 3) for i in range(1, 1+int(x.size(1)/self.stride))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x
'sigma_architecture2 is sigma without MLP'
class sigma_architecture2(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1, stride=None):
        super(sigma_architecture2, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.stride = stride
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        self.dense = nn.Linear(in_features=310, out_features=1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
        x = self.conv(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :self.stride*i, :], 3) for i in range(1, 1+int(x.size(1)/self.stride))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x
'sigma_architecture3 is sigma without convolutionsl layer and MLP'
class sigma_architecture3(nn.Module):
    def __init__(self, augment_include_time=True, T=1, stride=None):
        super(sigma_architecture3, self).__init__()
        self.augment_include_time = augment_include_time
        self.T = T
        self.stride = stride
        self.attention = layers.MultiHeadAttention(d_k=[2, 4, 8], in_features=[2, 4, 8], heads=3, out_features=14)
        self.dense = nn.Linear(in_features=28, out_features=1)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :self.stride*i, :], 3) for i in range(1, 1+int(x.size(1)/self.stride))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        return x

'''Example4: Varying Lengths'''
class cnn_length(nn.Module):
    def __init__(self):
        '''
        This CNN architecture is from https://doi.org/10.1080/14697688.2019.1654126
        Arg:
            - Input shape: (batch, channel, seq)
        '''
        super(cnn_length, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        self.dense1 = nn.LazyLinear(out_features=128)
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
class deepsignet_length(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(deepsignet_length, self).__init__()
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
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
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
class sigma_length(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1, stride=None):
        super(sigma_length, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.stride = stride
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        self.dense = fnn(input_size=310, hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
        x = self.conv(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :self.stride*i, :], 3) for i in range(1, 1+int(x.size(1)/self.stride))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        return x
class transformer_length(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1, grid_points=None):
        super(transformer_length, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=153, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        self.dense = fnn(input_size=int(grid_points*155), hidden_sizes=(32, 32, 32, 32, 32), output_size=1, activation=nn.ReLU())
        self.sigmoid = nn.Sigmoid()
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
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
        x = self.sigmoid(x)
        return x
class lstm_length(nn.Module):
    def __init__(self, grid_points=None):
        super(lstm_length, self).__init__()
        self.standardization = Standardization()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=2, batch_first=True)
        self.dense = fnn(input_size=int((grid_points-1)*128), hidden_sizes=(128, 64), output_size=1, activation=nn.PReLU())
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        x = self.standardization(x)
        x = torch.transpose(x, 1, 2)
        x = self.lstm(x)[0]
        x = self.flatten(x)
        x = self.dense(x)
        return x

'''Example5: Multiple'''
class cnn_multiple(nn.Module):
    def __init__(self):
        '''
        This CNN architecture is from https://doi.org/10.1080/14697688.2019.1654126
        Arg:
            - Input shape: (batch, channel, seq)
        '''
        super(cnn_multiple, self).__init__()
        self.conv1 = nn.Conv1d(in_channels=1, out_channels=32, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        self.conv2 = nn.Conv1d(in_channels=32, out_channels=64, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        self.conv3 = nn.Conv1d(in_channels=64, out_channels=128, kernel_size=20, stride=1, padding='same', padding_mode='zeros')
        self.dense1 = nn.LazyLinear(out_features=128)
        self.dense2 = nn.Linear(in_features=128, out_features=4)
        self.LRelu1 = nn.LeakyReLU(negative_slope=0.1)
        self.LRelu2 = nn.LeakyReLU(negative_slope=0.3)
        self.pooling = nn.MaxPool1d(kernel_size=3)
        self.dropout1 = nn.Dropout1d(p=0.25)
        self.dropout2 = nn.Dropout1d(p=0.4)
        self.dropout3 = nn.Dropout1d(p=0.3)
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
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
class deepsignet_multiple(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1):
        super(deepsignet_multiple, self).__init__()
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.dense = fnn(input_size=155, hidden_sizes=(32, 32, 32, 32, 32), output_size=4, activation=nn.ReLU())
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
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
class sigma_multiple(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1, stride=None):
        super(sigma_multiple, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.stride = stride
        self.sigmoid = nn.Sigmoid()
        self.conv = nn.Conv1d(in_channels=1, out_channels=3, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        self.dense = fnn(input_size=310, hidden_sizes=(32, 32, 32, 32, 32), output_size=4, activation=nn.ReLU())
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)
        self.linear_layer = nn.Linear(4, 4, bias=False)
        self.linear_layer.weight.data = torch.tensor([[1,0,0,0], [0,5,0,0], [0,0,1,0], [0,0,0,3]], dtype=torch.float32)
        self.linear_layer.weight.requires_grad = True

    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
            time = time.expand(x.shape[0], 1, x.shape[-1])
        x = self.conv(x)
        if self.augment_include_original is True:
            x = torch.cat((x, value), dim=1)
        if self.augment_include_time is True:
            x = torch.cat((x, time), dim=1)
        x = torch.transpose(x, 1, 2)
        x = torch.stack([layers.Sig(x[:, :self.stride*i, :], 3) for i in range(1, 1+int(x.size(1)/self.stride))])
        x = torch.transpose(x, 0, 1)
        x = self.attention(x)
        x = self.flatten(x)
        x = self.dense(x)
        x = self.sigmoid(x)
        x = self.linear_layer(x)
        return x
class transformer_multiple(nn.Module):
    def __init__(self, augment_include_original=True, augment_include_time=True, T=1, grid_points=None):
        super(transformer_multiple, self).__init__()
        self.augment_include_original = augment_include_original
        self.augment_include_time = augment_include_time
        self.T = T
        self.conv = nn.Conv1d(in_channels=1, out_channels=153, kernel_size=3, stride=1, padding='same', padding_mode='zeros')
        self.attention = layers.MultiHeadAttention(d_k=[5, 25, 125], in_features=[5, 25, 125], heads=3, out_features=155)
        self.dense = fnn(input_size=int(grid_points*155), hidden_sizes=(32, 32, 32, 32, 32), output_size=4, activation=nn.ReLU())
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        if self.augment_include_original is True:
            value = x
        if self.augment_include_time is True:
            time = torch.linspace(start=0, end=self.T, steps=x.shape[-1], device=x.device).view(1, 1, x.shape[-1])
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
class lstm_multiple(nn.Module):
    def __init__(self, grid_points=None):
        super(lstm_multiple, self).__init__()
        self.standardization = Standardization()
        self.lstm = nn.LSTM(input_size=1, hidden_size=128, num_layers=2, batch_first=True)
        self.dense = fnn(input_size=int((grid_points-1)*128), hidden_sizes=(128, 64), output_size=4, activation=nn.PReLU())
        self.flatten = nn.Flatten(start_dim=1, end_dim=-1)

    def forward(self, x):
        x = self.standardization(x)
        x = torch.transpose(x, 1, 2)
        x = self.lstm(x)[0]
        x = self.flatten(x)
        x = self.dense(x)
        return x