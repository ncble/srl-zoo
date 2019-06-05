from __future__ import print_function, division, absolute_import

import torch as th
import torch.nn as nn
import torch.nn.functional as F
import torchvision.models as models

from torchsummary import summary

class BaseModelSRL(nn.Module):
    """
    Base Class for a SRL network
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self, state_dim=2, img_shape=(3,224,224)):
        super(BaseModelSRL, self).__init__()
        # Do not define the attribute self.state_dim nor self.img_shape here.
        # e.g forward/inverse/reward models inherit also from BaseModelSRL
    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.forward(observations)

    def forward(self, x):
        raise NotImplementedError


class BaseModelAutoEncoder(BaseModelSRL):
    """
    Base Class for a SRL network (autoencoder family)
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self, state_dim=2, img_shape=(3,224,224)):
        super(BaseModelAutoEncoder, self).__init__(state_dim=state_dim, img_shape=img_shape)
        self.state_dim = state_dim
        self.img_shape = img_shape
        # Inspired by ResNet:
        # conv3x3 followed by BatchNorm2d
        self.encoder_conv = nn.Sequential(
            # 224x224xN_CHANNELS -> 112x112x64
            nn.Conv2d(self.img_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56x64

            conv3x3(in_planes=64, out_planes=64, stride=1),  # 56x56x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x64

            conv3x3(in_planes=64, out_planes=64, stride=2),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 6x6x64
        )

        self.decoder_conv = nn.Sequential(
            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 13x13x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 27x27x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 55x55x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, 64, kernel_size=3, stride=2),  # 111x111x64
            nn.BatchNorm2d(64),
            nn.ReLU(True),

            nn.ConvTranspose2d(64, self.img_shape[0], kernel_size=4, stride=2),  # 224x224xN_CHANNELS
        )

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.encode(observations)

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def forward(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.encode(x)


class BaseModelVAE(BaseModelAutoEncoder):
    """
    Base Class for a SRL network (VAE family)
    It implements a getState method to retrieve a state from observations
    """

    def __init__(self, state_dim=2, img_shape=(3,224,224)):
        super(BaseModelVAE, self).__init__(state_dim=state_dim, img_shape=img_shape)

    # def getStates(self, observations):
    #     """
    #     :param observations: (th.Tensor)
    #     :return: (th.Tensor)
    #     """
    #     return self.encode(observations)[0]

    def encode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def decode(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        raise NotImplementedError

    def reparameterize(self, mu, logvar):
        """
        Reparameterize for the backpropagation of z instead of q.
        (See "The reparameterization trick" section of https://arxiv.org/abs/1312.6114)
        :param mu: (th.Tensor)
        :param logvar: (th.Tensor)
        """
        if self.training:
            # logvar = \log(\sigma^2) = 2 * \log(\sigma)
            # \sigma = \exp(0.5 * logvar)
            std = logvar.mul(0.5).exp_()
            # Sample \epsilon from normal distribution
            # use std to create a new tensor, so we don't have to care
            # about running on GPU or not
            eps = std.new(std.size()).normal_()
            # Then multiply with the standard deviation and add the mean
            return eps.mul(std).add_(mu)
        else:
            return mu

    def forward(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.encode(x)[0]
    def compute_tensors(self, x):
        input_shape = x.size()
        mu, logvar = self.encode(x)
        z = self.reparameterize(mu, logvar)
        decoded = self.decode(z).view(input_shape)
        return decoded, mu, logvar


class CustomCNN(BaseModelSRL):
    """
    Convolutional Neural Network
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    """

    def __init__(self, state_dim=2, img_shape=(3,224,224)):
        super(CustomCNN, self).__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape
        # Inspired by ResNet:
        # conv3x3 followed by BatchNorm2d
        
        self.conv_layers = nn.Sequential(
            # 224x224x3 -> 112x112x64
            nn.Conv2d(self.img_shape[0], 64, kernel_size=7, stride=2, padding=3, bias=False),
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2, padding=1),  # 56x56x64

            conv3x3(in_planes=64, out_planes=64, stride=1),  # 56x56x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2),  # 27x27x64

            conv3x3(in_planes=64, out_planes=64, stride=2),  # 14x14x64
            nn.BatchNorm2d(64),
            nn.ReLU(inplace=True),
            nn.MaxPool2d(kernel_size=3, stride=2)  # 6x6x64
        )
        
        outshape = summary(self.conv_layers, img_shape, show=False) # [-1, channels, high, width]
        self.img_height, self.img_width = outshape[-2:]
        self.fc = nn.Linear(self.img_height * self.img_width * 64, state_dim)

    def forward(self, x):
        x = self.conv_layers(x)
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


def conv3x3(in_planes, out_planes, stride=1):
    """"
    From PyTorch Resnet implementation
    3x3 convolution with padding
    :param in_planes: (int)
    :param out_planes: (int)
    :param stride: (int)
    """
    return nn.Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=1, bias=False)


def encodeOneHot(tensor, n_dim):
    """
    One hot encoding for a given tensor
    :param tensor: (th Tensor)
    :param n_dim: (int) Number of dimensions
    :return: (th.Tensor)
    """
    encoded_tensor = th.Tensor(tensor.shape[0], n_dim).zero_().to(tensor.device)
    return encoded_tensor.scatter_(1, tensor.data, 1.)
class GaussianNoise(nn.Module):
    """
    Gaussian Noise layer
    :param batch_size: (int)
    :param input_dim: (int)
    :param std: (float) standard deviation
    :param mean: (float)
    :param device: (pytorch device)
    """

    def __init__(self, batch_size, input_dim, device, std, mean=0):
        super(GaussianNoise, self).__init__()
        self.std = std
        self.mean = mean
        self.device = device
        self.noise = th.zeros(batch_size, input_dim, device=self.device)

    def forward(self, x):
        if self.training:
            self.noise.data.normal_(self.mean, std=self.std)
            return x + self.noise
        return x


class GaussianNoiseVariant(nn.Module):
    """
    Variant of the Gaussian Noise layer that does not require fixed batch_size
    It recreates a tensor at each call
    :param device: (pytorch device)
    :param std: (float) standard deviation
    :param mean: (float)
    """

    def __init__(self, device, std, mean=0):
        super(GaussianNoiseVariant, self).__init__()
        self.std = std
        self.mean = mean
        self.device = device

    def forward(self, x):
        if self.training:
            noise = th.zeros(x.size(), device=self.device)
            noise.data.normal_(self.mean, std=self.std)
            return x + noise
        return x
if __name__ == "__main__":
    print("Start")

    img_shape = (3,128,128)
    model = CustomCNN(state_dim=2, img_shape=img_shape)
    A = summary(model, img_shape)
    


