from __future__ import print_function, division, absolute_import
import torch
import torch.nn as nn
import torch.nn.functional as F  # HACK TODO DEBUG
import numpy as np
from torchsummary import summary
try:
    # relative import: when executing as a package: python -m ...
    from .base_models import BaseModelAutoEncoder, ConvSN2d, ConvTransposeSN2d, LinearSN, UNet
    from ..losses.losses import autoEncoderLoss, AEboundLoss
    from .base_trainer import BaseTrainer
    from .gan import GeneratorUnet, EncoderUnet
except:
    # absolute import: when executing directly: python train.py ...
    from models.base_models import BaseModelAutoEncoder, ConvSN2d, ConvTransposeSN2d, LinearSN, UNet
    from losses.losses import autoEncoderLoss, AEboundLoss
    from models.base_trainer import BaseTrainer
    from models.gan import GeneratorUnet, EncoderUnet


class LinearAutoEncoder(BaseModelAutoEncoder):
    """
    :param state_dim: (int)
    :param img_shape: (tuple)
    """

    def __init__(self, state_dim, img_shape):
        super(LinearAutoEncoder, self).__init__(state_dim, img_shape)
        # BaseModelAutoEncoder.__init__(self)
        self.img_shape = img_shape

        self.encoder = nn.Sequential(
            nn.Linear(np.prod(self.img_shape), state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, np.prod(self.img_shape)),
            nn.Tanh()
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
        super(DenseAutoEncoder, self).__init__(state_dim, img_shape)
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
            nn.Tanh()
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

    def __init__(self, state_dim=3, img_shape=(3, 224, 224)):
        # state_dim=state_dim, img_shape=img_shape
        super(CNNAutoEncoder, self).__init__(state_dim=state_dim, img_shape=img_shape)
        self.state_dim = state_dim
        self.img_shape = img_shape

        outshape = summary(self.encoder_conv, img_shape, show=False)  # [-1, channels, high, width]
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


class UNetAutoEncoder(BaseModelAutoEncoder):
    """
    Custom UNet autoencoder network
    :param state_dim: (int)
    """

    def __init__(self, state_dim=3, img_shape=(3, 224, 224)):
        super(UNetAutoEncoder, self).__init__(state_dim=state_dim, img_shape=img_shape)
        self.decoder = GeneratorUnet(state_dim, img_shape, unet_bn=True)
        self.encoder = EncoderUnet(state_dim, img_shape, unet_bn=True)

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
    def __init__(self, state_dim=2, img_shape=(3, 224, 224)):
        super().__init__()
        # BaseTrainer.__init__(self)
        self.state_dim = state_dim
        self.img_shape = img_shape

    def build_model(self, model_type='custom_cnn'):
        assert model_type in ['custom_cnn', 'linear', 'mlp', 'unet']
        if model_type == 'custom_cnn':
            self.model = CNNAutoEncoder(self.state_dim, self.img_shape)
        elif model_type == 'mlp':
            self.model = DenseAutoEncoder(self.state_dim, self.img_shape)
        elif model_type == 'linear':
            self.model = LinearAutoEncoder(self.state_dim, self.img_shape)
        elif model_type == 'unet':
            self.model = UNetAutoEncoder(self.state_dim, self.img_shape)
        else:
            raise NotImplementedError("model type: ({}) not supported yet.".format(model_type))

    def train_on_batch(self, obs, next_obs, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        state_pred = self.encode(obs)
        decoded_obs = self.decode(state_pred)
        decoded_next_obs = self.reconstruct(next_obs)
        autoEncoderLoss(obs, decoded_obs, next_obs, decoded_next_obs, weight=1.0, loss_manager=loss_manager)
        AEboundLoss(state_pred, weight=1.0, loss_manager=loss_manager, name='bonud_state_loss', max_val=50)
        loss = self.update_nn_weights(optimizer, loss_manager, valid_mode=valid_mode)
        return loss

    def reconstruct(self, x):
        return self.model.decode(self.model.encode(x))

    def encode(self, x):
        return self.model.encode(x)

    def decode(self, x):
        return self.model.decode(x)

    def forward(self, x):
        return self.model.encode(x)  # or self.model(x)


if __name__ == "__main__":
    print("Start")
    from torchsummary import summary
    img_shape = (3, 128, 128)
    model = CNNAutoEncoder(state_dim=2, img_shape=img_shape)
    A = summary(model, img_shape)
    # import ipdb; ipdb.set_trace()
