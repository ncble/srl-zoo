
import os
import glob
import sys
import numpy as np

# Adapted from https://discuss.pytorch.org/t/unet-implementation/426
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm


def ConvSN2d(in_channels, out_channels, kernel_size,
             stride=1,
             padding=0,
             dilation=1,
             groups=1,
             bias=True,
             padding_mode='zeros'):
    A = spectral_norm(nn.Conv2d(in_channels, out_channels,
                                kernel_size=kernel_size,
                                stride=stride,
                                padding=padding,
                                dilation=dilation,
                                groups=groups,
                                bias=bias))
    # A.__class__.__name__ = 'ConvSN2d' ## [TODO]
    return A


def ConvTransposeSN2d(in_channels, out_channels, kernel_size,
                      stride=1,
                      padding=0,
                      output_padding=0,
                      groups=1,
                      bias=True,
                      dilation=1,
                      padding_mode='zeros'):
    A = spectral_norm(nn.ConvTranspose2d(in_channels, out_channels,
                                         kernel_size=kernel_size,
                                         stride=stride,
                                         padding=padding,
                                         output_padding=output_padding,
                                         groups=groups,
                                         bias=bias,
                                         dilation=dilation))
    # A.__class__.__name__ = 'ConvTransposeSN2d' ## [TODO]
    return A


def LinearSN(in_features, out_features, bias=True):
    A = spectral_norm(nn.Linear(in_features, out_features, bias=bias))
    # A.__class__.__name__ = 'LinearSN' ## [TODO]
    return A


class UNet(nn.Module):
    def __init__(
        self,
        in_ch=3,
        out_ch=3,
        depth=4,
        start_ch=64,
        inc_rate=2,
        padding=True,
        batch_norm=True,
        spec_norm=False,
        dropout=0.5,
        up_mode='upconv',
        include_top=True
    ):
        """
        Implementation of
        U-Net: Convolutional Networks for Biomedical Image Segmentation
        (Ronneberger et al., 2015)
        https://arxiv.org/abs/1505.04597
        Args:
            in_ch (int): number of input channels
            out_ch (int): number of output channels
            depth (int): depth of the network
            start_ch (int): number of filters in the first layer is start_ch
            padding (bool): if True, apply padding such that the input shape
                            is the same as the output.
                            This may introduce artifacts
            batch_norm (bool): Use BatchNorm after layers with an
                               activation function
            dropout (None or float): Use dropout (if not None) in Conv block.
            up_mode (str): one of 'upconv' or 'upsample'.
                           'upconv' will use transposed convolutions for
                           learned upsampling.
                           'upsample' will use bilinear upsampling.
        """
        super(UNet, self).__init__()
        assert up_mode in ('upconv', 'upsample')
        self.include_top = include_top
        self.padding = padding
        self.out_ch = out_ch
        self.depth = depth
        self.spec_norm = spec_norm
        prev_channels = in_ch
        self.down_path = nn.ModuleList()
        for i in range(depth+1):
            self.down_path.append(
                UNetConvBlock(prev_channels, (inc_rate ** i) * start_ch,
                              padding, batch_norm, dropout, spec_norm=self.spec_norm)
            )
            prev_channels = (inc_rate ** i) * start_ch

        self.up_path = nn.ModuleList()
        for i in reversed(range(depth)):
            self.up_path.append(
                UNetUpBlock(prev_channels, (inc_rate ** i) * start_ch, up_mode,
                            padding, batch_norm, dropout, spec_norm=self.spec_norm)
            )
            prev_channels = (inc_rate ** i) * start_ch

        if self.include_top:
            if self.spec_norm:
                self.last = ConvSN2d(prev_channels, out_ch, kernel_size=1)
            else:
                self.last = nn.Conv2d(prev_channels, out_ch, kernel_size=1)
        else:
            self.out_ch = prev_channels
        self.tanh_act = nn.Tanh()

    def forward(self, x):
        blocks = []
        for i, down in enumerate(self.down_path):
            x = down(x)
            if i != len(self.down_path) - 1:
                blocks.append(x)
                x = F.max_pool2d(x, 2)

        for i, up in enumerate(self.up_path):
            x = up(x, blocks[-i - 1])

        if self.include_top:
            x = self.last(x)
            return self.tanh_act(x)
        else:
            return x


class UNetConvBlock(nn.Module):
    def __init__(self, in_size, out_size, padding, batch_norm, dropout, spec_norm=False):
        super(UNetConvBlock, self).__init__()
        self.spec_norm = spec_norm
        block = []
        if self.spec_norm:
            # [stride=1] padding = (k-1)/2
            block.append(ConvSN2d(in_size, out_size,
                                  kernel_size=3, padding=int(padding)))
        else:
            # [stride=1] padding = (k-1)/2
            block.append(nn.Conv2d(in_size, out_size,
                                   kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))
        if dropout is not None:
            block.append(nn.Dropout(p=dropout))
        if self.spec_norm:
            block.append(ConvSN2d(out_size, out_size,
                                  kernel_size=3, padding=int(padding)))
        else:
            block.append(nn.Conv2d(out_size, out_size,
                                   kernel_size=3, padding=int(padding)))
        block.append(nn.ReLU())
        if batch_norm:
            block.append(nn.BatchNorm2d(out_size))

        self.block = nn.Sequential(*block)

    def forward(self, x):
        out = self.block(x)
        return out


class UNetUpBlock(nn.Module):
    def __init__(self, in_size, out_size, up_mode, padding, batch_norm, dropout, spec_norm=False):
        super(UNetUpBlock, self).__init__()
        self.padding = padding
        self.spec_norm = spec_norm
        if up_mode == 'upconv':
            if self.spec_norm:
                self.up = ConvTransposeSN2d(
                    in_size, out_size, kernel_size=3, stride=2, padding=1, output_padding=1)
            else:
                # self.up = nn.ConvTranspose2d(in_size, out_size, kernel_size=2, stride=2)
                self.up = nn.ConvTranspose2d(
                    in_size, out_size, kernel_size=3, stride=2, padding=1, output_padding=1)
        elif up_mode == 'upsample':

            if self.spec_norm:
                self.up = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2),
                    ConvSN2d(in_size, out_size, kernel_size=1),
                )
            else:
                self.up = nn.Sequential(
                    nn.Upsample(mode='bilinear', scale_factor=2),
                    nn.Conv2d(in_size, out_size, kernel_size=1),
                )
        self.relu_act = nn.ReLU()
        self.conv_block = UNetConvBlock(
            in_size, out_size, padding, batch_norm, dropout, spec_norm=self.spec_norm)

    def center_crop(self, layer, target_size):
        _, _, layer_height, layer_width = layer.size()
        diff_y = (layer_height - target_size[0]) // 2
        diff_x = (layer_width - target_size[1]) // 2
        return layer[:, :, diff_y: (diff_y + target_size[0]), diff_x: (diff_x + target_size[1])]

    def forward(self, x, bridge):
        up = self.up(x)
        up = self.relu_act(up)
        # print("Brige shape: {}, target size: {}".format(bridge.shape, up.shape[2:]))
        if self.padding:
            crop1 = bridge
        else:
            crop1 = self.center_crop(bridge, up.shape[2:])
        # print(up.shape)
        # print(crop1.shape)
        out = torch.cat([up, crop1], 1)
        out = self.conv_block(out)

        return out


class Generator(nn.Module):
    def __init__(self, state_dim, img_shape,
                 unet_depth=3,
                 unet_ch=32,
                 spectral_norm=False,
                 unet_bn=False,
                 unet_drop=0.0):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.spectral_norm = spectral_norm
        self.unet_depth = unet_depth
        self.unet_ch = unet_ch
        self.unet_drop = unet_drop
        self.unet_bn = unet_bn
        # self.lipschitz_G = 1.1 [TODO]
        assert self.img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        if self.spectral_norm:
            # state_layer = DenseSN(np.prod(self.img_shape), activation=None, lipschitz=self.lipschitz_G)(state_input)
            self.first = LinearSN(
                self.state_dim, np.prod(self.img_shape), bias=True)
        else:
            self.first = nn.Linear(
                self.state_dim, np.prod(self.img_shape), bias=True)

            # state_layer = Dense(np.prod(self.img_shape), activation=None)(state_input)
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU(negative_slope=0.2)],
            ['prelu', nn.PReLU()],
            ['tanh', nn.Tanh()],
            ['relu', nn.ReLU()]
        ])

        out_channels = self.img_shape[0]  # = 3
        in_channels = out_channels
        self.unet = UNet(in_ch=in_channels, include_top=False, depth=self.unet_depth, start_ch=self.unet_ch,
                         batch_norm=self.unet_bn, spec_norm=self.spectral_norm, dropout=self.unet_drop, up_mode='upconv', out_ch=out_channels)
        prev_channels = self.unet.out_ch
        if self.spectral_norm:
            self.last = ConvSN2d(prev_channels, out_channels,
                                 kernel_size=3, stride=1, padding=1)
        else:
            self.last = nn.Conv2d(
                prev_channels, out_channels, kernel_size=3, stride=1, padding=1)

    def forward(self, x):
        x = self.first(x)
        x = self.activations['lrelu'](x)
        x = x.view(x.size(0), *self.img_shape)
        x = self.unet(x)
        x = self.last(x)
        x = self.activations['tanh'](x)
        return x


class Discriminator(nn.Module):
    def __init__(self, img_shape, state_dim,
                 spectral_norm=False,
                 d_chs=32):
        super().__init__()
        self.img_shape = img_shape
        self.state_dim = state_dim
        self.spectral_norm = spectral_norm
        self.d_chs = d_chs

        assert self.img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU(negative_slope=0.2)],
            ['prelu', nn.PReLU()],
            ['tanh', nn.Tanh()],
            ['relu', nn.ReLU()]
        ])
        self.modules_list = nn.ModuleList([])

        def d_layer(prev_channels, out_channels, kernel_size=4, spectral_norm=False):
            """Discriminator layer"""
            if spectral_norm:
                # [stride=2] padding = (kernel_size/2) -1
                layer = ConvSN2d(prev_channels, out_channels,
                               kernel_size=kernel_size, stride=2, padding=1)
            else:
                # [stride=2] padding = (kernel_size/2) -1
                layer = nn.Conv2d(prev_channels, out_channels,
                                kernel_size=kernel_size, stride=2, padding=1)
            return [layer, self.activations['lrelu']]#, out.out_channels

        start_chs = self.img_shape[0]
        self.modules_list.extend(d_layer(start_chs, self.d_chs, spectral_norm=self.spectral_norm))
        self.modules_list.extend(d_layer(self.d_chs, self.d_chs*2, spectral_norm=self.spectral_norm))
        self.modules_list.extend(d_layer(self.d_chs*2, self.d_chs*4, spectral_norm=self.spectral_norm))
        self.modules_list.extend(d_layer(self.d_chs*4, self.d_chs*8, spectral_norm=self.spectral_norm))
        self.modules_list.extend(d_layer(self.d_chs*8, self.d_chs*8, spectral_norm=self.spectral_norm))
        if self.spectral_norm:
            self.modules_list.append(ConvSN2d(self.d_chs*8, self.d_chs*4,
                               kernel_size=3, stride=1, padding=1))
            in_features = self.modules_list[-1]
            self.before_last = LinearSN(in_features, self.state_dim, bias=True)
        else:
            self.modules_list.append(nn.Conv2d(self.d_chs*8, self.d_chs*4,
                               kernel_size=3, stride=1, padding=1))
            self.before_last = nn.Linear(in_features, self.state_dim, bias=True)
        
                            
    def forward(self, x):
        for layer in self.modules_list:
            x = layer(x)
        x = self.before_last(x)
        x = self.activations['lrelu'](x)
        x = x.view(x.size(0), -1) ## flatten
        x = self.last(x)
        x = self.activations['tanh'](x)

        if spectral_norm:
            d_out = ConvSN2D(self.d_chs*4, kernel_size=3,
                             strides=1, padding='same')(d_out)
        else:
            d_out = Conv2D(self.d_chs*4, kernel_size=3,
                           strides=1, padding='same')(d_out)
        d_out = LeakyReLU(alpha=0.2)(d_out)
        if self.normalize_D:
            d_out = InstanceNormalization()(d_out)

        # validity = Dense(1, activation=None)(Flatten()(d_out))

        # validity = Dense(1, activation='sigmoid')(Flatten()(d_out))

        d_out = Flatten()(d_out)
        # d_out = Dropout(rate=0.3)(d_out)

        # d_out = Dense(self.state_dim//3, activation='relu')(d_out) # Exp_201 and before
        if spectral_norm:
            d_out = DenseSN(self.state_dim, activation='relu')(d_out)
        else:
            d_out = Dense(self.state_dim, activation='relu')(
                d_out)  # Exp_202 and after

        if self.loss_name == 'ns_gan':
            validity = Dense(1, activation='sigmoid')((d_out))
        elif self.loss_name == 'wgan_gp':
            validity = Dense(1, activation=None)((d_out))
        elif self.loss_name == 'ns_gan_sn':
            validity = DenseSN(1, activation='sigmoid')((d_out))
        else:
            raise ValueError("Not implemented error.")

        return x


if __name__ == "__main__":
    print("Start")
    # device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
    from torchsummary import summary  # requires torchsummary==1.5.2,
    # device = torch.device("cuda:0")
    # chs = 3
    # model = UNet(in_ch=chs, out_ch=3, start_ch=32, depth=3, batch_norm=False,
    #              spec_norm=False)  # .to(device)
    # a = 128
    # summary(model, (chs, a, a))

    model = Generator(10, img_shape=(3, 128, 128))
    summary(model, (10,))
