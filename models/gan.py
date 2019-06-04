import numpy as np
import torch
from torch import nn
import torch.nn.functional as F
try:
    # relative import
    from .base_models import BaseModelSRL, ConvSN2d, ConvTransposeSN2d, LinearSN, UNet
    from .base_trainer import BaseTrainer
    from ..losses.losses import ganNonSaturateLoss, autoEncoderLoss, ganBCEaccuracy, AEboundLoss
except:
    from models.base_models import BaseModelSRL, ConvSN2d, ConvTransposeSN2d, LinearSN, UNet
    from models.base_trainer import BaseTrainer
    from losses.losses import ganNonSaturateLoss, autoEncoderLoss, ganBCEaccuracy, AEboundLoss
    
class Generator(nn.Module):
    def __init__(self, state_dim, img_shape,
                 unet_depth=2, # 3
                 unet_ch=16, # 32
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
    def __init__(self, state_dim, img_shape,
                 spectral_norm=False,
                 d_chs=16): # 32
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

    def __init__(self, state_dim, img_shape,
                 unet_depth=2, # 3 
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
        self.encoder = Encoder(self.state_dim, self.img_shape, spectral_norm=False)
        self.generator = Generator(self.state_dim, self.img_shape, spectral_norm=True)
        self.discriminator = Discriminator(self.state_dim, self.img_shape, spectral_norm=True)

    def forward(self, x):
        return self.encoder(x)
    def train_on_batch_E(self, obs, next_obs, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        """
        loss_manager will cumulate the loss (pytorch tensor) of e.g. inverse/forward/reward models, etc.

        """
        ## Compute loss tensor (add to the previous losses cumulated in loss_manager)
        # state_pred, state_pred_next = self.encoder(obs), self.encoder(next_obs)
        # reconstruct_obs, reconstruct_obs_next = self.generator(state_pred), self.generator(state_pred_next)
        
        # reconstruct_obs, reconstruct_obs_next = self.reconstruct(obs), self.reconstruct(next_obs)

        state_pred = self.encoder(obs)
        reconstruct_obs = self.generator(state_pred)
        reconstruct_obs_next = self.reconstruct(next_obs)
        autoEncoderLoss(obs, reconstruct_obs, next_obs, reconstruct_obs_next, 10000.0, loss_manager)
        AEboundLoss(state_pred, 1.0, loss_manager) ## NEW [TODO]
        loss = self.update_nn_weights(optimizer, loss_manager, valid_mode=valid_mode)
        return loss
    def train_on_batch_D(self, obs, label_valid, label_fake, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        sample_state = torch.randn((obs.size(0), self.state_dim), requires_grad=False).to(device)
        fake_img = self.generator(sample_state)
        # fake_loss 
        fake_img_rating = self.discriminator(fake_img.detach())
        ganNonSaturateLoss(fake_img_rating, label_fake, weight=1.0, loss_manager=loss_manager, name="ns_loss_D_fake")
        # real_loss
        real_img_rating = self.discriminator(obs)
        ganNonSaturateLoss(real_img_rating, label_valid, weight=1.0, loss_manager=loss_manager, name="ns_loss_D_real")
        loss = self.update_nn_weights(optimizer, loss_manager, valid_mode=valid_mode)
        acc_pos = ganBCEaccuracy(real_img_rating, label=1)
        acc_neg = ganBCEaccuracy(fake_img_rating, label=0)
        acc = (acc_pos + acc_neg) / 2.
        return loss, acc.item()

    def train_on_batch_G(self, obs, label_valid, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        sample_state = torch.randn((obs.size(0), self.state_dim), requires_grad=False).to(device)
        fake_img = self.generator(sample_state)
        fake_rating = self.discriminator(fake_img)
        ganNonSaturateLoss(fake_rating, label_valid, weight=1.0, loss_manager=loss_manager, name="ns_loss_G")
        loss = self.update_nn_weights(optimizer, loss_manager, valid_mode=valid_mode)
        acc = ganBCEaccuracy(fake_rating, label=1)
        return loss, acc.item()
    def reconstruct(self, x):
        return self.generator(self.encoder(x))





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

    # model = Generator(10, img_shape=(3, 128, 128))
    # summary(model, (10,))

    # model = Discriminator(4, (3, 128, 128))
    # summary(model, (3, 128, 128))

    model = Encoder(4, (3, 128, 128))
    summary(model, (3, 128, 128))
