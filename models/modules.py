from .autoencoders import AutoEncoderTrainer
from .vae import VAETrainer#CNNVAE, DenseVAE
from .forward_inverse import BaseForwardModel, BaseInverseModel, BaseRewardModel
from .priors import SRLConvolutionalNetwork, SRLDenseNetwork, SRLLinear
from .triplet import EmbeddingNet
from .gan import GANTrainer # Generator, Discriminator, Encoder, UNet, 
try:
    ## relative import: when executing as a package: python -m ...
    from ..losses.losses import forwardModelLoss, inverseModelLoss, rewardModelLoss
    from .base_trainer import BaseTrainer
    from ..utils import printRed
    from ..preprocessing.preprocess import N_CHANNELS
except:
    ## absolute import: when executing directly: python train.py ...
    from losses.losses import forwardModelLoss, inverseModelLoss, rewardModelLoss
    from models.base_trainer import BaseTrainer
    from utils import printRed
    from preprocessing.preprocess import N_CHANNELS


class SRLModules(BaseForwardModel, BaseInverseModel, BaseRewardModel):
    def __init__(self, state_dim=2, img_shape=None, action_dim=6, cuda=False, model_type="custom_cnn", losses=None,
                 inverse_model_type="linear"):
        """
        A model that can combine AE/VAE + Inverse + Forward + Reward models
        :param state_dim: (int)
        :param img_shape: (tuple or None) channels first ! 
        :param action_dim: (int)
        :param cuda: (bool)
        :param model_type: (str)
        :param losses: ([str])
        :param inverse_model_type: (str) Architecture of the inverse model ('linear', 'mlp')
        """
        self.model_type = model_type
        self.losses = losses
        BaseForwardModel.__init__(self)
        BaseInverseModel.__init__(self)
        BaseRewardModel.__init__(self)

        self.cuda = cuda
        if img_shape is None:
            self.img_shape = (N_CHANNELS, 224, 224)
        else:
            if N_CHANNELS != img_shape[0]:
                print("="*50)
                printRed("Warning: N_CHANNELS={} is inconsistent with argument img_shape={}.".format(N_CHANNELS, img_shape[0]))
                print("="*50)
                self.img_shape = (N_CHANNELS, ) + img_shape[1:]
            else:
                self.img_shape = img_shape
        self.initForwardNet(state_dim, action_dim)
        self.initInverseNet(state_dim, action_dim, model_type=inverse_model_type)
        self.initRewardNet(state_dim)
        def getInputDim():
            return np.prod(self.img_shape)
        # Architecture
        if "autoencoder" in losses or "dae" in losses:
            self.model = AutoEncoderTrainer(state_dim=state_dim, img_shape=self.img_shape)
            self.model.build_model(model_type=model_type)
        elif "vae" in losses:
            self.model = VAETrainer(state_dim=state_dim, img_shape=self.img_shape)
            self.model.build_model(model_type=model_type)
        else:
            pass
        if model_type == 'gan':
            self.model = GANTrainer(img_shape=self.img_shape, state_dim=state_dim)
            self.model.build_model()
            
        # if model_type == "custom_cnn":
        #     if "autoencoder" in losses or "dae" in losses:
        #         self.model = CNNAutoEncoder(state_dim, img_shape=self.img_shape)
        #     else:
        #         # for losses not depending on specific architecture (supervised, inv, fwd..)
        #         self.model = CustomCNN(state_dim, img_shape=self.img_shape)

        # elif model_type == "mlp":
        #     if "autoencoder" in losses or "dae" in losses:
        #         self.model = DenseAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
        #     elif "vae" in losses:
        #         self.model = DenseVAE(input_dim=getInputDim(),
        #                               state_dim=state_dim)
        #     else:
        #         # for losses not depending on specific architecture (supervised, inv, fwd..)
        #         self.model = SRLDenseNetwork(getInputDim(), state_dim, cuda=cuda)

        # elif model_type == "linear":
        #     if "autoencoder" in losses or "dae" in losses:
        #         self.model = LinearAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
        #     else:
        #         # for losses not depending on specific architecture (supervised, inv, fwd..)
        #         self.model = SRLLinear(input_dim=getInputDim(), state_dim=state_dim, cuda=cuda)

        # elif model_type == "resnet":
        #     self.model = SRLConvolutionalNetwork(state_dim, cuda)
        # elif model_type == "gan":
        #     self.model = Encoder(self.img_shape, state_dim, spectral_norm=False)
        #     self.generator = Generator(self.img_shape, state_dim, spectral_norm=True)
        #     self.discriminator = Discriminator(self.img_shape, state_dim, spectral_norm=True)
        
        # # elif: [Add new model here !]

        # if losses is not None and "triplet" in losses:
        #     # pretrained resnet18 with fixed weights
        #     self.model = EmbeddingNet(state_dim)

    def forward(self, x):
        if self.model_type == 'linear' or self.model_type == 'mlp':
            x = x.contiguous()
        return self.model(x)

    def encode(self, x):
        if "triplet" in self.losses:
            return self.model(x)
        else:
            raise NotImplementedError()

    def forwardTriplets(self, anchor, positive, negative):
        """
        Overriding the forward function in the case of Triplet loss
        anchor : anchor observations (th. Tensor)
        positive : positive observations (th. Tensor)
        negative : negative observations (th. Tensor)
        """
        return self.model(anchor), self.model(positive), self.model(negative)
    
    def add_forward_loss(self, states, actions_st, next_states, loss_manager):
        next_states_pred = self.forwardModel(states, actions_st)
        forwardModelLoss(next_states_pred, next_states, weight=1.0, loss_manager=loss_manager)

    def add_inverse_loss(self, states, actions_st, next_states, loss_manager):
        actions_pred = self.inverseModel(states, next_states)
        inverseModelLoss(actions_pred, actions_st, weight=1.0, loss_manager=loss_manager)

    def add_reward_loss(self, states, rewards_st, next_states, loss_manager):
        rewards_pred = self.rewardModel(states, next_states)
        rewardModelLoss(rewards_pred, rewards_st, weight=1.0, loss_manager=loss_manager)

class SRLModulesSplit(BaseForwardModel, BaseInverseModel, BaseRewardModel):
    def __init__(self, state_dim=2, action_dim=6, cuda=False, model_type="custom_cnn",
                 losses=None, split_dimensions=None, n_hidden_reward=16, inverse_model_type="linear"):
        """
        A model that can split representation, combining
        AE/VAE for the first split with Inverse + Forward in the second split
        Reward model is learned for all the dimensions
        :param state_dim: (int)
        :param action_dim: (int)
        :param cuda: (bool)
        :param model_type: (str)
        :param losses: ([str])
        :param split_dimensions: (OrderedDict) Number of dimensions for the different losses
        :param n_hidden_reward: (int) Number of hidden units for the reward model
        :param inverse_model_type: (str) Architecture of the inverse model ('linear', 'mlp')
        """
        assert len(split_dimensions) == len(losses), "Please specify as many split dimensions {} as losses {} !". \
            format(len(split_dimensions), len(losses))

        n_dims = sum(split_dimensions.values())
        # Account for shared dimensions
        n_dims += list(split_dimensions.values()).count(-1)
        assert n_dims == state_dim, \
            "The sum of all splits' dimensions {} must be equal to the state dimension {}"\
            .format(sum(split_dimensions.values()), str(state_dim))

        self.split_dimensions = split_dimensions
        self.model_type = model_type
        self.losses = losses

        BaseForwardModel.__init__(self)
        BaseInverseModel.__init__(self)
        BaseRewardModel.__init__(self)

        self.cuda = cuda
        self.state_dim = state_dim

        self.initForwardNet(self.state_dim, action_dim)
        self.initInverseNet(self.state_dim, action_dim, model_type=inverse_model_type)
        self.initRewardNet(self.state_dim, n_hidden=n_hidden_reward)

        # Architecture
        if model_type == "custom_cnn":
            if "autoencoder" in losses or "dae" in losses:
                self.model = CNNAutoEncoder(state_dim)
            elif "vae" in losses:
                self.model = CNNVAE(state_dim)
            else:
                self.model = CustomCNN(state_dim)

        elif model_type == "mlp":
            if "autoencoder" in losses or "dae" in losses:
                self.model = DenseAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
            elif "vae" in losses:
                self.model = DenseVAE(input_dim=getInputDim(), state_dim=state_dim)
            else:
                self.model = SRLDenseNetwork(getInputDim(), state_dim, cuda=cuda)

        elif model_type == "linear":
            if "autoencoder" in losses:
                self.model = LinearAutoEncoder(input_dim=getInputDim(), state_dim=state_dim)
            else:
                self.model = SRLLinear(input_dim=getInputDim(), state_dim=state_dim, cuda=cuda)

        elif model_type == "resnet":
            # No support because of autoencoder
            raise ValueError("Resnet not supported when splitting representation")

        if "triplet" in losses:
            raise ValueError("triplet not supported when splitting representation")

    def getStates(self, observations):
        """
        :param observations: (th.Tensor)
        :return: (th.Tensor)
        """
        return self.model.getStates(observations)

    def forward(self, x):
        if "autoencoder" in self.losses or "dae" in self.losses:
            return self.forwardAutoencoder(x)
        elif "vae" in self.losses:
            return self.forwardVAE(x)
        else:
            return self.model.forward(x)

    def detachSplit(self, tensor, index):
        """
        Detach splits from the graph,
        so no gradients are backpropagated
        for those splits part of the states
        :param tensor: (th.Tensor)
        :param index (str) name of the split not to detach
        :return: (th.Tensor)
        """
        tensors = []
        start_idx = 0
        pred_dim = 0

        for key, n_dim in self.split_dimensions.items():
            n_dim = int(n_dim)
            # dealing with a split shared with the previous loss dimensions
            if n_dim == -1 and start_idx > 0:
                if key != index:
                    # Skip current index because it shares
                    # its dimensions with previous index
                    continue
                n_dim = 0
                # retrieving the previous index
                start_idx -= pred_dim

            if key != index:
                # tensors.append(tensor[:, start_idx:start_idx + n_dim].detach())
                # Mask state dimensions
                tensors.append(th.zeros_like(tensor[:, start_idx:start_idx + n_dim]))
            else:
                if n_dim == 0:
                    # Keeping the dimensions share with the previous loss/split attached
                    tensors[-1] = tensor[:, start_idx:start_idx + pred_dim]
                else:
                    tensors.append(tensor[:, start_idx:start_idx + n_dim])

            # Update previous dimension only if needed
            if n_dim > 0:
                pred_dim = n_dim
                # updating the index & storing dimensions of the previous loss/split
                start_idx += n_dim
            else:
                # Restore the start index
                start_idx += pred_dim

        return th.cat(tensors, dim=1)

    def forwardVAE(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        mu, logvar = self.model.encode(x)
        z = self.model.reparameterize(self.detachSplit(mu, index='vae'),
                                      self.detachSplit(logvar, index='vae'))
        decoded = self.model.decode(z).view(input_shape)
        return decoded, self.detachSplit(mu, index='vae'), self.detachSplit(logvar, index='vae')

    def forwardAutoencoder(self, x):
        """
        :param x: (th.Tensor)
        :return: (th.Tensor)
        """
        input_shape = x.size()
        encoded = self.model.encode(x)
        decoded = self.model.decode(self.detachSplit(encoded, index='autoencoder')).view(input_shape)
        return encoded, decoded

    def inverseModel(self, state, next_state):
        """
        Predict action given current state and next state
        :param state: (th.Tensor)
        :param next_state: (th.Tensor)
        :return: probability of each action
        """
        return self.inverse_net(th.cat((self.detachSplit(state, index='inverse'),
                                        self.detachSplit(next_state, index='inverse')), dim=1))

    def forwardModel(self, state, action):
        """
        Predict next state given current state and action
        :param state: (th.Tensor)
        :param action: (th Tensor)
        :return: (th.Tensor)
        """
        # Predict the delta between the next state and current state
        concat = th.cat((self.detachSplit(state, index='forward'),
                         encodeOneHot(action, self.action_dim)), dim=1)
        return self.detachSplit(state, index='forward') + self.forward_net(concat)

    def rewardModel(self, state, next_state):
        """
        Predict reward given current state and next state
        :param state: (th.Tensor)
        :param next_state: (th Tensor)
        :return: (th.Tensor)
        """
        return self.reward_net(th.cat((self.detachSplit(state, index='reward'),
                                       self.detachSplit(next_state, index='reward')), dim=1))
