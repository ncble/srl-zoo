
import os
import glob
import sys
import numpy as np

# Adapted from https://discuss.pytorch.org/t/unet-implementation/426
import torch
from torch import nn
import torch.nn.functional as F
from torch.nn.utils import spectral_norm
try:
    # relative import
    from .models import BaseModelSRL
    from .base_trainer import BaseTrainer
except:
    from models.models import BaseModelSRL
    from models.base_trainer import BaseTrainer

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
    def __init__(self, img_shape, state_dim,
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
            ['relu', nn.ReLU()],
            ['sigmoid', nn.Sigmoid()]
        ])
        self.modules_list = nn.ModuleList([])
        COUNT_IMG_REDUCE = 0

        def d_layer(prev_channels, out_channels, kernel_size=4, spectral_norm=False):
            """Discriminator layer"""
            nonlocal COUNT_IMG_REDUCE
            COUNT_IMG_REDUCE += 1
            if spectral_norm:
                # [stride=2] padding = (kernel_size/2) -1
                layer = ConvSN2d(prev_channels, out_channels,
                                 kernel_size=kernel_size, stride=2, padding=1)
            else:
                # [stride=2] padding = (kernel_size/2) -1
                layer = nn.Conv2d(prev_channels, out_channels,
                                  kernel_size=kernel_size, stride=2, padding=1)
            return [layer, self.activations['lrelu']]  # , out.out_channels

        start_chs = self.img_shape[0]
        self.modules_list.extend(
            d_layer(start_chs, self.d_chs, spectral_norm=self.spectral_norm))
        self.modules_list.extend(
            d_layer(self.d_chs, self.d_chs*2, spectral_norm=self.spectral_norm))
        self.modules_list.extend(
            d_layer(self.d_chs*2, self.d_chs*4, spectral_norm=self.spectral_norm))
        self.modules_list.extend(
            d_layer(self.d_chs*4, self.d_chs*8, spectral_norm=self.spectral_norm))
        self.modules_list.extend(
            d_layer(self.d_chs*8, self.d_chs*8, spectral_norm=self.spectral_norm))

        if self.spectral_norm:
            self.modules_list.append(ConvSN2d(self.d_chs*8, self.d_chs*4,
                                              kernel_size=3, stride=1, padding=1))

            last_channels = self.modules_list[-1].out_channels
            times = COUNT_IMG_REDUCE
            in_features = last_channels * \
                (self.img_shape[1]//2**times) * (self.img_shape[2]//2**times)
            self.before_last = LinearSN(in_features, self.state_dim, bias=True)
            self.last = LinearSN(self.state_dim, 1, bias=True)
        else:
            self.modules_list.append(nn.Conv2d(self.d_chs*8, self.d_chs*4,
                                               kernel_size=3, stride=1, padding=1))
            last_channels = self.modules_list[-1].out_channels
            times = COUNT_IMG_REDUCE
            in_features = last_channels * \
                (self.img_shape[1]//2**times) * (self.img_shape[2]//2**times)
            self.before_last = nn.Linear(
                in_features, self.state_dim, bias=True)
            self.last = nn.Linear(self.state_dim, 1, bias=True)

    def forward(self, x):
        for layer in self.modules_list:
            x = layer(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.activations['lrelu'](x)
        x = self.before_last(x)
        x = self.activations['lrelu'](x)
        x = self.last(x)
        x = self.activations['sigmoid'](x)
        return x

class Encoder(BaseModelSRL):
    """
    
    Note: Only Encoder has getStates method.
    """
    def __init__(self, img_shape, state_dim,
                 unet_depth=3,
                 unet_ch=16,
                 unet_bn=False,
                 unet_drop=0.0,
                 spectral_norm=False):
        super().__init__()
        assert img_shape[0] < 10, "Pytorch uses 'channel first' convention."
        self.state_dim = state_dim
        self.img_shape = img_shape
        self.spectral_norm = spectral_norm
        self.unet_depth = unet_depth
        self.unet_ch = unet_ch
        self.unet_drop = unet_drop
        self.unet_bn = unet_bn
        self.activations = nn.ModuleDict([
            ['lrelu', nn.LeakyReLU(negative_slope=0.2)],
            ['prelu', nn.PReLU()],
            ['tanh', nn.Tanh()],
            ['relu', nn.ReLU()],
            ['sigmoid', nn.Sigmoid()]
        ])
        self.unet = UNet(in_ch=self.img_shape[0], include_top=False, depth=self.unet_depth, start_ch=self.unet_ch,
                         batch_norm=self.unet_bn, spec_norm=self.spectral_norm, dropout=self.unet_drop, up_mode='upconv', out_ch=1)
        prev_channels = self.unet.out_ch

        self.modules_list = nn.ModuleList([])

        if self.spectral_norm:
            inter_features = 1 * (self.img_shape[1]//2**2) * (self.img_shape[2]//2**2)
            self.modules_list.append(ConvSN2d(prev_channels, 1, kernel_size=4, stride=2, padding=1))
            self.modules_list.append(self.activations['lrelu'])
            self.modules_list.append(ConvSN2d(1, 1, kernel_size=4, stride=2, padding=1))
            self.modules_list.append(self.activations['lrelu'])
            self.before_last = LinearSN(inter_features, 100, bias=True)
            self.last = LinearSN(100, self.state_dim, bias=True)
        else:
            inter_features = 1 * (self.img_shape[1]//2**2) * (self.img_shape[2]//2**2)
            self.modules_list.append(nn.Conv2d(prev_channels, 1, kernel_size=4, stride=2, padding=1))
            self.modules_list.append(self.activations['lrelu'])
            self.modules_list.append(nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1))
            self.modules_list.append(self.activations['lrelu'])
            self.before_last = nn.Linear(inter_features, 100, bias=True)
            self.last = nn.Linear(100, self.state_dim, bias=True)
            # self.top_model = nn.Sequential(OrderDict([
            #                     ('conv1', nn.Conv2d(prev_channels, 1, kernel_size=4, stride=2, padding=1)),
            #                     ('relu1', self.activations['relu']),
            #                     ('conv2', nn.Conv2d(1, 1, kernel_size=4, stride=2, padding=1)),
            #                     ('relu2', self.activations['relu']),
            #                     ('dense1', nn.Linear(inter_features, 100, bias=True)),
            #                     ('relu3', self.activations['relu']),
            #                     ('dense2', nn.Linear(100, self.state_dim, bias=True)),
            #                 ]))


    def forward(self, x):
        x = self.unet(x)
        for layer in self.modules_list:
            x = layer(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.before_last(x)
        x = self.activations['lrelu'](x)
        x = self.last(x)
        return x
    


class GANTrainer(BaseTrainer):
    def __init__(self, img_shape=None, state_dim=2):
        super().__init__()
        self.img_shape = img_shape
        self.state_dim = state_dim
    def build_model(self):
        self.encoder = Encoder(self.img_shape, self.state_dim, spectral_norm=False)
        self.generator = Generator(self.img_shape, self.state_dim, spectral_norm=True)
        self.discriminator = Discriminator(self.img_shape, self.state_dim, spectral_norm=True)

    def forward(self, x):
        return self.encoder(x)
    def train_on_batch_E(self, obs, next_obs, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        """
        loss_manager will cumulate the loss (pytorch tensor) of e.g. inverse/forward/reward models, etc.

        """
        ## Compute loss tensor (add to the previous losses cumulated in loss_manager)
        state_pred = self.encoder(obs)
        reconstruct_obs = self.generator(state_pred)
        state_pred_next = self.encoder(next_obs)
        reconstruct_obs_next = self.generator(state_pred_next)
        autoEncoderLoss(obs, reconstruct_obs, next_obs, reconstruct_obs_next, 10000.0, loss_manager)
        loss = self.update_nn_weights(optimizer, loss_manager, valid_mode=valid_mode)
        return loss
    def train_on_batch_D(self, obs, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        label_valid = torch.ones((obs.size(0), 1)).to(device)
        label_fake = torch.zeros((obs.size(0), 1)).to(device)
        sample_state = torch.randn((obs.size(0), self.state_dim),
                                    requires_grad=False).to(device)
        fake_img = self.generator(sample_state)
        # fake_loss 
        ganNonSaturateLoss(self.model.discriminator(fake_img.detach()), label_fake, weight=1.0, loss_manager=loss_manager, name="ns_loss_D_fake")
        # real_loss
        ganNonSaturateLoss(self.model.discriminator(obs), label_valid, weight=1.0, loss_manager=loss_manager, name="ns_loss_D_real")
        loss = self.update_nn_weights(optimizer, loss_manager, valid_mode=valid_mode)
        return loss

    def train_on_batch_G(self, obs, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        sample_state = torch.randn((obs.size(0), self.state_dim),
                        requires_grad=False).to(device)
        fake_img = self.model.generator(sample_state)
        fake_rating = self.model.discriminator(fake_img)
        ganNonSaturateLoss(fake_rating, label_valid, weight=1.0, loss_manager=loss_manager, name="ns_loss_G")
        loss = self.update_nn_weights(optimizer, loss_manager, valid_mode=valid_mode)
        return loss
    
    def train_on_batch(self, obs, next_obs, optimizers, loss_managers, valid_mode=False, device=torch.device('cpu')):
        raise NotImplementedError
        self.train_on_batch_E(obs, next_obs, optimizers[0], loss_managers[0], valid_mode=valid_mode, device=device)
        self.train_on_batch_D(obs, optimizers[1], loss_managers[1], valid_mode=valid_mode, device=device)
        self.train_on_batch_G(obs, optimizers[2], loss_managers[2], valid_mode=valid_mode, device=device)
        return 




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

    # model = Generator(img_shape=(3, 128, 128), 10)
    # summary(model, (10,))

    # model = Discriminator((3, 128, 128), 4)
    # summary(model, (3, 128, 128))

    model = Encoder((3,128,128), 4)
    summary(model, (3, 128, 128))
