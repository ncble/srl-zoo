from __future__ import print_function, division, absolute_import

from torch.autograd import Function

from .base_models import *
import torch.nn.functional as F


class SRLConvolutionalNetwork(BaseModelSRL):
    """
    Convolutional Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    :param img_shape: (tuple) (3, H, W)
    :param noise_std: (float)  To avoid NaN (states must be different)
    """

    def __init__(self, state_dim=2, img_shape=(3, 224, 224), noise_std=1e-6):
        super(SRLConvolutionalNetwork, self).__init__(state_dim=state_dim, img_shape=img_shape)
        self.resnet = models.resnet18(pretrained=False)

        # Replace the last fully-connected layer
        n_units = self.resnet.fc.in_features
        print("{} units in the last layer".format(n_units))

        self.resnet.fc = nn.Sequential(
            nn.Linear(n_units, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, state_dim),
        )
        # This variant does not require the batch_size
        self.noise = GaussianNoiseVariant(torch.device("cuda"), noise_std)  # [TODO, device]
        # self.noise = GaussianNoise(batch_size, state_dim, torch.device("cuda"), noise_std)

    def forward(self, x):
        x = self.resnet(x)
        if self.training:
            x = self.noise(x)
        return x


class SRLCustomCNN(BaseModelSRL):
    """
    Convolutional Neural Network for State Representation Learning
    input shape : 3-channel RGB images of shape (3 x H x W), where H and W are expected to be at least 224
    :param state_dim: (int)
    :param img_shape: (tuple) (3, H, W)
    :param noise_std: (float)  To avoid NaN (states must be different)
    """

    def __init__(self, state_dim=2, img_shape=(3, 224, 224), noise_std=1e-6):
        super(SRLCustomCNN, self).__init__(state_dim=state_dim, img_shape=img_shape)
        self.cnn = CustomCNN(state_dim)
        self.noise = GaussianNoiseVariant(torch.device("cuda"), noise_std)  # [TODO, device]

    def forward(self, x):
        x = self.cnn(x)
        if self.training:
            x = self.noise(x)
        return x


class SRLDenseNetwork(BaseModelSRL):
    """
    Dense Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W) (to be consistent with CNN network)
    :param state_dim: (int)
    :param img_shape: (tuple) (3, H, W)
    :param noise_std: (float)  To avoid NaN (states must be different)
    :param n_hidden: (int)
    """

    def __init__(self, state_dim=2, img_shape=(3, 224, 224),
                 n_hidden=64, noise_std=1e-6):
        super(SRLDenseNetwork, self).__init__(state_dim=state_dim, img_shape=img_shape)

        self.fc = nn.Sequential(
            nn.Linear(np.prod(img_shape), n_hidden),
            nn.ReLU(),
            nn.Linear(n_hidden, state_dim)
        )
        self.fc = self.fc
        self.noise = GaussianNoiseVariant(torch.device("cuda"), noise_std)  # [TODO, device]

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        if self.training:
            x = self.noise(x)
        return x


class SRLLinear(BaseModelSRL):
    """
    Dense Neural Net for State Representation Learning (SRL)
    input shape : 3-channel RGB images of shape (3 x H x W) (to be consistent with CNN network)
    :param state_dim: (int)
    :param img_shape: (tuple) (3, H, W)
    """

    def __init__(self, state_dim=2, img_shape=(3, 224, 224)):
        super(SRLLinear, self).__init__(state_dim=state_dim, img_shape=img_shape)

        self.fc = nn.Linear(np.prod(img_shape), state_dim)
        self.fc = self.fc

    def forward(self, x):
        # Flatten input
        x = x.view(x.size(0), -1)
        x = self.fc(x)
        return x


# From https://github.com/fungtion/DANN
class ReverseLayerF(Function):
    """
    Fonction to backpropagate the opposite of the gradient
    scaled by a constant
    """

    @staticmethod
    def forward(ctx, x, lambda_):
        """
        :param x: (th.Tensor)
        :param lambda_: (float) scaling factor
        :return: (th.Tensor)
        """
        ctx.lambda_ = lambda_
        # Equivalent to return x ?
        return x.view_as(x)

    @staticmethod
    def backward(ctx, grad_output):
        """
        :param grad_output: (th.Tensor)
        :return: (th.Tensor, None)
        """
        # Compute the opposite of the gradient
        output = grad_output.neg() * ctx.lambda_
        return output, None


class Discriminator(nn.Module):
    """
    Discriminator network to distinguish states from two different episodes
    :input_dim: (int) input_dim = 2 * state_dim
    """

    def __init__(self, input_dim):
        super(Discriminator, self).__init__()
        self.net = nn.Sequential(
            nn.Linear(input_dim, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 64),
            nn.ReLU(inplace=True),
            nn.Linear(64, 1),
            nn.Sigmoid()
        )

    def forward(self, x):
        return self.net(x)


if __name__ == "__main__":
    print("Start")

    img_shape = (3, 128, 128)
    model = SRLConvolutionalNetwork(state_dim=2, cuda=False)
    A = summary(model, img_shape)
