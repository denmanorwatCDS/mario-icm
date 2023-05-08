import torch.nn
from torch import nn

class ConvBlock(nn.Module):
    def __init__(self, in_channels, out_channels, batch_norm, skip_conn, consecutive_convs, activation):
        super().__init__()
        layers = nn.ModuleList()
        self.skip_conn = skip_conn
        for conv in range(consecutive_convs):
            conv_stride = 1
            conv_in_channels = out_channels
            if conv == 0:
                conv_in_channels = in_channels
            if conv == consecutive_convs-1:
                conv_stride = 2
            layers.append(nn.Conv2d(in_channels=conv_in_channels, out_channels=out_channels, kernel_size=3,
                                       stride=conv_stride, padding=1))
            layers.append(activation)
            if batch_norm:
                layers.append(nn.BatchNorm2d())
        self.long_conv = torch.nn.Sequential(*layers)


    def forward(self, x):
        if self.skip_conn:
            downsampler = nn.MaxPool2d(kernel_size=2, stride=2)
            downsampled_x = downsampler(x)
            return downsampled_x+self.long_conv(x)
        return self.long_conv(x)

class FeatureNet(nn.Module):
    def __init__(self, obs_shape, batch_norm=False, skip_conn=False, consecutive_convs=1, activation=nn.ELU,
                 total_blocks=4, feature_map_size=32):
        super().__init__()
        layers = nn.ModuleList()
        for block in range(total_blocks):
            if block == 0:
                in_channels = obs_shape[0]
            else:
                in_channels = feature_map_size
            layers.append(ConvBlock(in_channels, out_channels=feature_map_size, batch_norm=batch_norm,
                                    skip_conn=skip_conn, consecutive_convs=consecutive_convs, activation=activation()))
        self.model = nn.Sequential(*layers)
        with torch.no_grad():
            self.output_shape = self.model.forward(torch.zeros(obs_shape)).shape

    def forward(self, input):
        return self.model(input)

    def get_output_shape(self):
        return self.output_shape

class InverseNet(nn.Module):
    def __init__(self, group, bottleneck, input_shape, output_features, fc_qty):
        super().__init__()
        self.layers = nn.ModuleList()
        in_channels = input_shape[0]*2
        out_channels = in_channels//2
        input_features = torch.tensor(input_shape).prod()*2
        self.input_shape = input_shape
        #print(input_shape)
        #print("Input features to inverse net: {}".format(input_features))
        if group:
            self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1,
                                         groups=out_channels))
            self.layers.append(nn.ELU())
            in_channels, out_channels, input_features = in_channels//2, out_channels//2, input_features//2
            #print("Input features to inverse net after grouping: {}".format(input_features))
        if bottleneck:
            self.layers.append(nn.Conv2d(in_channels=in_channels, out_channels=out_channels, kernel_size=1, stride=1))
            self.layers.append(nn.ELU())
            input_features = input_features//2
            #print("Input features to inverse net after bottlenecking: {}".format(input_features))
        self.layers.append(nn.Flatten())
        #print("Input features to linear layer: {}".format(input_features))
        for i in range(fc_qty-1):
            self.layers.append(nn.Linear(input_features, input_features//2))
            self.layers.append(nn.ELU())
            input_features = input_features//2
        self.layers.append(nn.Linear(input_features, output_features))
        self.inverse_net = nn.Sequential(*self.layers)

    def forward(self, input):
        permuted_input = torch.zeros(input.shape).to(input.get_device())
        permuted_input[:, ::2, :, :] = input[:, :self.input_shape[0], :, :]
        permuted_input[:, 1::2, :, :] = input[:, self.input_shape[0]:, :, :]
        return self.inverse_net(permuted_input)