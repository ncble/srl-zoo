from __future__ import print_function, division, absolute_import

from .models import *
from torchsummary import summary

class LinearAutoEncoder(BaseModelAutoEncoder):
    """
    :param input_dim: (int)
    :param state_dim: (int)
    """

    def __init__(self, input_dim, state_dim=3):
        super(LinearAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, state_dim),
        )

        self.decoder = nn.Sequential(
            nn.Linear(state_dim, input_dim),
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
        return self.decoder(x)


class DenseAutoEncoder(BaseModelAutoEncoder):
    """
    Dense autoencoder network
    Known issue: it reconstructs the image but omits the robot arm
    :param input_dim: (int)
    :param state_dim: (int)
    """

    def __init__(self, input_dim, state_dim=3):
        super(DenseAutoEncoder, self).__init__()

        self.encoder = nn.Sequential(
            nn.Linear(input_dim, 50),
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
            nn.Linear(50, input_dim),
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
        return self.decoder(x)


class CNNAutoEncoder(BaseModelAutoEncoder):
    """
    Custom convolutional autoencoder network
    Input dim (same as ResNet): 3x224x224
    :param state_dim: (int)
    """

    def __init__(self, state_dim=3, img_shape=(3,224,224)):
        super(CNNAutoEncoder, self).__init__()
        outshape = summary(self.encoder_conv, img_shape, show=False) # [-1, channels, high, width]
        self.img_high, self.img_width = outshape[-2:]
        
        self.encoder_fc = nn.Sequential(
            nn.Linear(self.img_high * self.img_width * 64, state_dim)
        )

        self.decoder_fc = nn.Sequential(
            nn.Linear(state_dim, self.img_high * self.img_width * 64)
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
        decoded = decoded.view(x.size(0), 64, self.img_high, self.img_width)
        return self.decoder_conv(decoded)

if __name__ == "__main__":
    print("Start")
    from torchsummary import summary
    img_shape = (3,128,128)
    model = CNNAutoEncoder(state_dim=2, img_shape=img_shape)
    A = summary(model, img_shape)
    # import ipdb; ipdb.set_trace()