from __future__ import print_function, division, absolute_import

from .models import BaseModelAutoEncoder
from torchsummary import summary
import torch
import torch.nn as nn
import torch.nn.functional as F ## HACK TODO DEBUG
import numpy as np
try:
    ## relative import: when executing as a package: python -m ...
    from ..losses.losses import autoEncoderLoss
    from .base_trainer import BaseTrainer
except:
    ## absolute import: when executing directly: python train.py ...
    from losses.losses import autoEncoderLoss
    from models.base_trainer import BaseTrainer

class LinearAutoEncoder(BaseModelAutoEncoder):
    """
    :param state_dim: (int)
    :param img_shape: (tuple)
    """

    def __init__(self, state_dim, img_shape):
        super(LinearAutoEncoder, self).__init__()
        # BaseModelAutoEncoder.__init__(self)
        self.img_shape = img_shape
        
        self.encoder = nn.Sequential(
            nn.Linear(np.prod(self.img_shape), state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, np.prod(self.img_shape)),
        )

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        x = self.decoder(x)
        x = x.view(x.size(0), *self.img_shape)
        return x


class DenseAutoEncoder(BaseModelAutoEncoder):
    """
    Dense autoencoder network
    Known issue: it reconstructs the image but omits the robot arm
    :param state_dim: (int)
    :param img_shape: (tuple)
    """

    def __init__(self, state_dim, img_shape):
        super(DenseAutoEncoder, self).__init__()
        # BaseModelAutoEncoder.__init__(self)
        self.img_shape = img_shape
        self.encoder = nn.Sequential(
            nn.Linear(np.prod(self.img_shape), 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, 50),
            nn.Tanh(),
            nn.Linear(50, 50),
            nn.Tanh(),
            nn.Linear(50, np.prod(self.img_shape)),
        )

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        # Flatten input
        x = x.view(x.size(0), -1)
        return self.encoder(x)

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        x = self.decoder(x)
        x = x.view(x.size(0), *self.img_shape)
        return x


class CNNAutoEncoder(BaseModelAutoEncoder):
    """
    Custom convolutional autoencoder network
    Input dim (same as ResNet): 3x224x224
    :param state_dim: (int)
    """

    def __init__(self, state_dim=3, img_shape=(3,224,224)):
        # state_dim=state_dim, img_shape=img_shape
        super(CNNAutoEncoder, self).__init__()
        # BaseModelAutoEncoder.__init__(self)
        outshape = summary(self.encoder_conv, img_shape, show=False) # [-1, channels, high, width]
        self.img_height, self.img_width = outshape[-2:]
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.img_height * self.img_width * 64, state_dim)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim, self.img_height * self.img_width * 64)
        )

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        encoded = self.encoder_conv(x)
        encoded = encoded.view(encoded.size(0), -1)
        return self.encoder_fc(encoded)

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        decoded = self.decoder_fc(x)
        decoded = decoded.view(x.size(0), 64, self.img_height, self.img_width)
        return self.decoder_conv(decoded)


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

class Encoder(nn.Module):
    """
    
    Note: Only Encoder has getStates method.
    """

    def __init__(self, img_shape, state_dim,
                 unet_depth=2,  # 3
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
    def forward(self, x):
        x = self.unet(x)
        for layer in self.modules_list:
            x = layer(x)
        x = x.view(x.size(0), -1)  # flatten
        x = self.before_last(x)
        x = self.activations['lrelu'](x)
        x = self.last(x)
        return x


class Generator(nn.Module):
    def __init__(self, img_shape, state_dim,
                 unet_depth=2,  # 3
                 unet_ch=16,  # 32
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


class UNetAutoEncoder(BaseModelAutoEncoder):
    """
    Custom UNet autoencoder network
    :param state_dim: (int)
    """

    def __init__(self, state_dim=3, img_shape=(3, 224, 224)):
        # state_dim=state_dim, img_shape=img_shape
        super(UNetAutoEncoder, self).__init__()
        # BaseModelAutoEncoder.__init__(self)
        self.decoder = Generator(img_shape, state_dim, unet_bn=True)
        self.encoder = Encoder(img_shape, state_dim, unet_bn=True)
        

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.encoder(x)

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.decoder(x)

class AutoEncoderTrainer(BaseTrainer):
    def __init__(self, state_dim=2, img_shape=(3,224,224)):
        super().__init__()
        # BaseTrainer.__init__(self)
        self.state_dim = state_dim
        self.img_shape = img_shape

    def build_model(self, model_type='custom_cnn'):
        assert model_type in ['custom_cnn', 'linear', 'mlp', 'unet']
        if model_type == 'custom_cnn':
            self.model = CNNAutoEncoder(self.state_dim, self.img_shape)
            # CNNAutoEncoder.__init__(self, self.state_dim, self.img_shape)
            # super(CNNAutoEncoder, self).__init__(
            #     state_dim=self.state_dim, img_shape=self.img_shape)
            # super().CNNAutoEncoder(state_dim=self.state_dim, img_shape=self.img_shape)
        elif model_type == 'mlp':
            self.model = DenseAutoEncoder(self.state_dim, self.img_shape)
            # DenseAutoEncoder.__init__(self, self.state_dim, np.prod(self.img_shape))
        elif model_type == 'linear':
            self.model = LinearAutoEncoder(self.state_dim, self.img_shape)
            # LinearAutoEncoder.__init__(self, self.state_dim, np.prod(self.img_shape))
        elif model_type == 'unet':
            self.model = UNetAutoEncoder(self.state_dim, self.img_shape)
        else:
            raise NotImplementedError("model type: ({}) not supported yet.".format(model_type))


    def train_on_batch(self, obs, next_obs, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        decoded_obs = self.reconstruct(obs)
        decoded_next_obs = self.reconstruct(next_obs)
        autoEncoderLoss(obs, decoded_obs, next_obs, decoded_next_obs, weight=1.0, loss_manager=loss_manager)
        loss = self.update_nn_weights(optimizer, loss_manager, valid_mode=valid_mode)
        return loss

    def reconstruct(self, x):
        return self.model.decode(self.model.encode(x))
    def encode(self, x):
        return self.model.encode(x)
    def decode(self, x):
        return self.model.decode(x)
    def forward(self, x):
        return self.model.encode(x) ## or self.model(x)

if __name__ == "__main__":
    print("Start")
    from torchsummary import summary
    img_shape = (3,128,128)
    model = CNNAutoEncoder(state_dim=2, img_shape=img_shape)
    A = summary(model, img_shape)
    # import ipdb; ipdb.set_trace()
