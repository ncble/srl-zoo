from __future__ import print_function, division, absolute_import

import os
import json
import sys
import time
from collections import defaultdict, OrderedDict
from pprint import pprint


import numpy as np
import torch
from torchvision.utils import make_grid
import torch.utils.data
from sklearn.utils import shuffle as sk_shuffle
from tqdm import tqdm

from losses.losses import LossManager, autoEncoderLoss, roboticPriorsLoss, tripletLoss, rewardModelLoss, \
    rewardPriorLoss, forwardModelLoss, inverseModelLoss, episodePriorLoss, l1Loss, l2Loss, kullbackLeiblerLoss, \
    perceptualSimilarityLoss, generationLoss, ganNonSaturateLoss
from losses.utils import findPriorsPairs
from pipeline import NAN_ERROR
from plotting.representation_plot import plotRepresentation, plt, plotImage
from preprocessing.data_loader import DataLoader, RobotEnvDataset
from preprocessing.utils import deNormalize
from utils import printRed, detachToNumpy, printYellow
from .modules import SRLModules, SRLModulesSplit
from .priors import Discriminator as PriorDiscriminator


MAX_BATCH_SIZE_GPU = 256  # For plotting, max batch_size before having memory issues
EPOCH_FLAG = 1  # Plot every 1 epoch
ITER_FLAG = 10 # Print loss every 10 iterations
N_WORKERS = 10

# The following variables are defined using arguments of the main script train.py
# SAVE_PLOTS = True
BATCH_SIZE = 256
N_EPOCHS = 1
VALIDATION_SIZE = 0.2  # 20% of training data for validation
# Experimental: episode independent prior
# Whether to do Uniform (default) or balanced sampling
BALANCED_SAMPLING = False


class BaseLearner(object):
    """
    Base class for a method that learn a state representation
    from observations
    :param state_dim: (int)
    :param batch_size: (int)
    :param seed: (int)
    :param cuda: (bool)
    """

    def __init__(self, state_dim, batch_size, seed=1, cuda=False):
        super(BaseLearner, self).__init__()
        self.state_dim = state_dim
        self.batch_size = batch_size
        self.model = None
        self.seed = seed
        self.use_dae = False
        # Seed the random generator
        np.random.seed(seed)
        torch.manual_seed(seed)
        if cuda: # This will increase the training by 5-10%.
            # Make CuDNN Determinist
            torch.backends.cudnn.deterministic = True
            torch.cuda.manual_seed(seed)

        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and cuda else "cpu")

    def _predFn(self, observations):
        """
        Predict states in test mode given observations

        :param observations: (torch.Tensor)
        :return: (np.ndarray)
        """
        # Move the tensor back to the cpu
        return detachToNumpy(self.model.getStates(observations))

    def predStatesWithDataLoader(self, data_loader):
        """
        Predict states using minibatches to avoid memory issues
        :param data_loader: (DataLoader object)
        :return: (np.ndarray)
        """
        predictions = []
        for obs_var in data_loader:
            obs_var = obs_var.to(self.device)
            predictions.append(self._predFn(obs_var))

        return np.concatenate(predictions, axis=0)

    def learn(self, *args, **kwargs):
        """
        Function called to learn a state representation
        it returns the learned states for the given observations
        """
        raise NotImplementedError("Learn method not implemented")

    @staticmethod
    def saveStates(states, images_path, rewards, log_folder, name=""):
        """
        Save learned states to json and npz files

        :param states: (np.ndarray)
        :param images_path: ([str])
        :param rewards: (rewards)
        :param log_folder: (str)
        :param name: (str)
        """
        print("Saving image path to state representation (image_to_state{}.json)".format(name))

        image_to_state = {path: list(map(str, state)) for path, state in zip(images_path, states)}

        with open("{}/image_to_state{}.json".format(log_folder, name), 'w') as f:
            json.dump(image_to_state, f, sort_keys=True)

        print("Saving states and rewards (states_rewards{}.npz)".format(name))

        states_rewards = {'states': states, 'rewards': rewards}
        np.savez('{}/states_rewards{}.npz'.format(log_folder, name), **states_rewards)


class SRL4robotics(BaseLearner):
    """
    Main Class for training a SRL model

    :param state_dim: (int)
    :param model_type: (str) one of "resnet", "mlp" or "custom_cnn"
    :param inverse_model_type: (str) one of "linear" or "mlp"
    :param log_folder: (str)
    :param seed: (int)
    :param learning_rate: (float)
    :param learning_rate_gan: tuple of float (lr_D, lr_G)
    :param l1_reg: (float) weight for l1 regularization
    :param l2_reg: (float) weight for l2 regularization
    :param cuda: (bool)
    :param multi_view: (bool)
    :param losses: ([str])
    :param losses_weights_dict: (OrderedDict)
    :param n_actions: (int)
    :param beta: (float) for beta-vae
    :param split_dimensions:
    :param path_to_dae: (str) path to pre-trained DAE when using perceptual loss
    :param state_dim_dae: (int)
    :param occlusion_percentage: (float) max percentage of occlusion when using DAE
    """

    def __init__(self, state_dim, img_shape=None, model_type="resnet", inverse_model_type="linear", log_folder="logs/default",
                 seed=1, learning_rate=0.001, learning_rate_gan=(None, None), l1_reg=0.0, l2_reg=0.0, cuda=False,
                 multi_view=False, losses=None, losses_weights_dict=None, n_actions=6, beta=1,
                 split_dimensions=-1, path_to_dae=None, state_dim_dae=200, occlusion_percentage=None):

        super(SRL4robotics, self).__init__(state_dim, BATCH_SIZE, seed, cuda)

        self.multi_view = multi_view
        self.losses = losses
        self.dim_action = n_actions
        self.beta = beta
        self.denoiser = None
        self.img_shape = img_shape
        self.model_type = model_type
        if model_type in ["linear", "mlp", "resnet", "custom_cnn", 'gan'] \
                or "autoencoder" in losses or "vae" in losses:
            self.use_forward_loss = "forward" in losses
            self.use_inverse_loss = "inverse" in losses
            self.use_reward_loss = "reward" in losses
            self.no_priors = "priors" not in losses
            self.episode_prior = "episode-prior" in losses
            self.reward_prior = "reward-prior" in losses
            self.use_autoencoder = "autoencoder" in losses
            self.use_vae = "vae" in losses
            self.use_triplets = "triplet" in self.losses
            self.perceptual_similarity_loss = "perceptual" in self.losses
            self.use_dae = "dae" in self.losses
            self.path_to_dae = path_to_dae

            if isinstance(split_dimensions, OrderedDict) and sum(split_dimensions.values()) > 0:
                printYellow("Using splitted representation")
                self.model = SRLModulesSplit(state_dim=self.state_dim, action_dim=self.dim_action,
                                             model_type=model_type, cuda=cuda, losses=losses,
                                             split_dimensions=split_dimensions, inverse_model_type=inverse_model_type)
            else:
                self.model = SRLModules(state_dim=self.state_dim, img_shape=self.img_shape, action_dim=self.dim_action, model_type=model_type,
                                        cuda=cuda, losses=losses, inverse_model_type=inverse_model_type)
        else:
            raise ValueError("Unknown model: {}".format(model_type))

        print("Using {} model".format(model_type))

        self.cuda = cuda
        self.device = torch.device(
            "cuda:0" if torch.cuda.is_available() and cuda else "cpu")

        if self.episode_prior:
            self.prior_discriminator = PriorDiscriminator(
                2 * self.state_dim).to(self.device)

        self.model = self.model.to(self.device)

        if self.model_type != 'gan':
            learnable_params = [param for param in self.model.parameters() if param.requires_grad]
            if self.episode_prior:
                learnable_params += [p for p in self.prior_discriminator.parameters()]
            self.optimizer = torch.optim.Adam(
                learnable_params, lr=learning_rate)
        else:
            assert not self.episode_prior, "NotImplementedError"
            self.optimizer_D = torch.optim.Adam(
                self.model.discriminator.parameters(), lr=learning_rate_gan[0], betas=(0.5, 0.9))
            self.optimizer_G = torch.optim.Adam(
                self.model.generator.parameters(), lr=learning_rate_gan[1], betas=(0.5, 0.9))
            combined_E_params = [
                param for param in self.model.model.parameters() if param.requires_grad]
            combined_E_params += [
                param for param in self.model.forward_net.parameters() if param.requires_grad]
            combined_E_params += [
                param for param in self.model.inverse_net.parameters() if param.requires_grad]
            combined_E_params += [
                param for param in self.model.reward_net.parameters() if param.requires_grad]
            self.optimizer_E = torch.optim.Adam(
                combined_E_params, lr=learning_rate, betas=(0.5, 0.9))
            # [TODO: check learnable parameters]
        self.log_folder = log_folder

        # Default weights that are updated with the weights passed to the script
        self.losses_weights_dict = {"forward": 1.0, "inverse": 2.0, "reward": 1.0, "priors": 1.0,
                                    "episode-prior": 1.0, "reward-prior": 10, "triplet": 1.0,
                                    "autoencoder": 1.0, "vae": 0.5e-6, "perceptual": 1e-6, "dae": 1.0,
                                    'l1_reg': l1_reg, "l2_reg": l2_reg, 'random': 1.0}
        self.occlusion_percentage = occlusion_percentage
        self.state_dim_dae = state_dim_dae

        if losses_weights_dict is not None:
            self.losses_weights_dict.update(losses_weights_dict)

        if self.use_dae and self.occlusion_percentage is not None:
            print("Using a maximum occlusion surface of {}".format(str(self.occlusion_percentage)))

    @staticmethod
    def loadSavedModel(log_folder, valid_models, cuda=True, img_shape=None):
        """
        Load a saved SRL model

        :param log_folder: (str)
        :param valid_models: ([str])
        :param cuda: (bool)
        :return: (SRL4robotics object, OrderedDict)
        """
        # Sanity checks
        assert os.path.exists(log_folder), "Error: folder '{}' does not exist".format(log_folder)
        assert os.path.exists(log_folder + "exp_config.json"), \
            "Error: could not find 'exp_config.json' in '{}'".format(log_folder)
        assert os.path.exists(log_folder + "srl_model.pth"), \
            "Error: could not find 'srl_model.pth' in '{}'".format(log_folder)

        with open(log_folder + 'exp_config.json', 'r') as f:
            # IMPORTANT: keep the order for the losses
            # so the json is loaded as an OrderedDict
            exp_config = json.load(f, object_pairs_hook=OrderedDict)

        state_dim = exp_config['state-dim']
        losses = exp_config['losses']
        n_actions = exp_config['n_actions']
        model_type = exp_config['model-type']
        multi_view = exp_config.get('multi-view', False)
        split_dimensions = exp_config.get('split-dimensions', -1)
        model_path = log_folder + 'srl_model.pth'
        inverse_model_type = exp_config.get('inverse-model-type', 'linear')
        occlusion_percentage = exp_config.get('occlusion-percentage', 0)

        difference = set(losses).symmetric_difference(valid_models)
        assert set(losses).intersection(valid_models) != set(), "Error: Not supported losses " + ", ".join(difference)

        srl_model = SRL4robotics(state_dim, img_shape=img_shape, model_type=model_type, cuda=cuda, multi_view=multi_view,
                                 losses=losses, n_actions=n_actions, split_dimensions=split_dimensions,
                                 inverse_model_type=inverse_model_type, occlusion_percentage=occlusion_percentage)
        srl_model.model.load_state_dict(torch.load(model_path))

        return srl_model, exp_config

    def learn(self, images_path, actions, rewards, episode_starts, figdir=None, monitor_mode='loss'):
        """
        Learn a state representation
        :param images_path: (numpy 1D array)
        :param actions: (np.ndarray)
        :param rewards: (numpy 1D array)
        :param episode_starts: (numpy 1D array) boolean array
                                the ith index is True if one episode starts at this frame
        :param figdir: directory path, save figures to the folder.
        :param monitor_mode: options are ['loss', 'pbar']
        :return: (np.ndarray) the learned states for the given observations
        """

        print("\nYour are using the following weights for the losses:")
        pprint(self.losses_weights_dict)
        assert monitor_mode in ['loss', 'pbar'], "monitor should be either 'loss' or 'prgressbar'"
        # PREPARE DATA -------------------------------------------------------------------------------------------------
        # here, we organize the data into minibatches
        # and find pairs for the respective loss terms (for robotics priors only)
        if figdir is not None:
            figdir_repr = os.path.join(figdir, "state_repr") # state representation scatter plot
            figdir_recon = os.path.join(figdir, "img_recon")  # image reconstruction folder
            os.makedirs(figdir_repr, exist_ok=True)
            os.makedirs(figdir_recon, exist_ok=True)
        
        data_loader_params = {'batch_size': self.batch_size,
                  'shuffle': True,
                  'num_workers': N_WORKERS,
                  'pin_memory': False}
        data_loader_params_test = {'batch_size': 128,
                  'shuffle': False,
                  'num_workers': N_WORKERS,
                  'pin_memory': False}
        sample_indices = np.arange(len(images_path))
        ## Shuffle datasets
        sample_indices, images_path, actions, rewards, episode_starts = sk_shuffle(sample_indices, images_path, actions, rewards, episode_starts, random_state=0)
        valid_size = np.round(VALIDATION_SIZE * len(images_path)).astype(np.int64)

        indices_train, imgspath_train, act_train, rew_train, epis_train = sample_indices[:-valid_size], images_path[:-valid_size], actions[:-valid_size], rewards[:-valid_size], episode_starts[:-valid_size]
        indices_val, imgspath_val, act_val, rew_val, epis_val           = sample_indices[-valid_size:], images_path[-valid_size:], actions[-valid_size:], rewards[-valid_size:], episode_starts[-valid_size:]
        
        train_set = RobotEnvDataset(indices_train, images_path, actions, rewards, episode_starts, 
                                 is_training=True, img_shape=self.img_shape, multi_view=self.multi_view,
                                 use_triplets=self.use_triplets, apply_occlusion=self.use_dae,
                                 occlusion_percentage=self.occlusion_percentage, dtype=np.float32)
        valid_set = RobotEnvDataset(indices_val, images_path, actions, rewards, episode_starts, 
                                 is_training=True, img_shape=self.img_shape, multi_view=self.multi_view,
                                 use_triplets=self.use_triplets, apply_occlusion=self.use_dae,
                                 occlusion_percentage=self.occlusion_percentage, dtype=np.float32)
        test_set = RobotEnvDataset(sample_indices, images_path, actions, rewards, episode_starts, 
                                 is_training=False, img_shape=self.img_shape, multi_view=self.multi_view,
                                 use_triplets=self.use_triplets, apply_occlusion=self.use_dae,
                                 occlusion_percentage=self.occlusion_percentage, dtype=np.float32)
        dataloader_train = torch.utils.data.DataLoader(train_set, **data_loader_params)
        dataloader_valid = torch.utils.data.DataLoader(valid_set, **data_loader_params)
        dataloader_test  = torch.utils.data.DataLoader(test_set, **data_loader_params_test)
        # ========================= Print some info =========================
        # Stats about actions
        action_set = set(actions)
        n_actions = int(np.max(actions) + 1)
        print("{} unique actions / {} actions".format(len(action_set), n_actions))
        n_pairs_per_action = np.zeros(n_actions, dtype=np.int64)
        n_obs_per_action = np.zeros(n_actions, dtype=np.int64)
        for i in range(n_actions):
            n_obs_per_action[i] = np.sum(actions == i)
        print("Number of observations per action")
        print(n_obs_per_action)
        print("Train: {} minibatches, {} samples".format(len(dataloader_train), len(train_set)))
        print("Valid: {} minibatches, {} samples".format(len(dataloader_valid), len(valid_set)))
        # =======================================================================

        dissimilar_pairs, same_actions_pairs = None, None
        if not self.no_priors:
            dissimilar_pairs, same_actions_pairs = findPriorsPairs(self.batch_size, minibatchlist, actions, rewards,
                                                                   n_actions, n_pairs_per_action)

        if self.use_vae and self.perceptual_similarity_loss and self.path_to_dae is not None:

            self.denoiser = SRLModules(state_dim=self.state_dim_dae, img_shape=self.img_shape, action_dim=self.dim_action,
                                       model_type="custom_cnn",
                                       cuda=self.cuda, losses=["dae"])
            self.denoiser.load_state_dict(torch.load(self.path_to_dae))
            self.denoiser.eval()
            self.denoiser = self.denoiser.to(self.device)
            for param in self.denoiser.parameters():
                param.requires_grad = False

        if self.episode_prior:
            idx_to_episode = {idx: episode_idx for idx, episode_idx in enumerate(np.cumsum(episode_starts))}
            minibatch_episodes = [[idx_to_episode[i] for i in minibatch] for minibatch in minibatchlist]

        
        # TRAINING -----------------------------------------------------------------------------------------------------
        loss_history = defaultdict(list)
        loss_manager = LossManager(self.model, loss_history)
        if self.model_type == 'gan':
            loss_history_D = defaultdict(list)
            loss_history_G = defaultdict(list)
            loss_manager_D = LossManager(self.model.discriminator, loss_history_D)
            loss_manager_G = LossManager(self.model.generator, loss_history_G)

        best_error = np.inf
        best_model_path = "{}/srl_model.pth".format(self.log_folder)
        

        # Random features, we don't need to train a model
        if len(self.losses) == 1 and self.losses[0] == 'random':
            global N_EPOCHS
            N_EPOCHS = 0
            printYellow("Skipping training because using random features")
            torch.save(self.model.state_dict(), best_model_path)
        

        for epoch in range(N_EPOCHS):
            for valid_mode, dataloader in enumerate([dataloader_train, dataloader_valid]): ## [TODO: lisibility!]
                # import ipdb; ipdb.set_trace()
                if monitor_mode == 'pbar':
                    pbar = tqdm(total=len(dataloader))
                if self.model_type == 'gan':
                    # GAN's training requires multi-optimizers, thus multiple loss_manger/epoch_loss/val_loss, etc.
                    epoch_loss_D, epoch_batches_D = 0, 0
                    epoch_loss_G, epoch_batches_G = 0, 0
                epoch_loss, epoch_batches = 0, 0
                n_batch_per_epoch = len(dataloader)

                start_time = time.time()
                if valid_mode:
                    self.model.eval()
                    # with torch.no_grad():
                    self.prev_mode = torch.is_grad_enabled()
                    torch.set_grad_enabled(False)
                    # torch._C.set_grad_enabled(False)
                else:
                    self.model.train()
                
                for iter_ind, (sample_idx, obs, next_obs, action, reward, noisy_obs, next_noisy_obs) in enumerate(dataloader):
                    obs, next_obs = obs.to(self.device), next_obs.to(self.device)
                    if self.use_dae:
                        noisy_obs = noisy_obs.to(self.device)
                        next_noisy_obs = next_noisy_obs.to(self.device)
                    
                    if self.model_type == 'gan':
                        # GAN's training requires multi-optimizers.
                        self.optimizer_D.zero_grad()
                        self.optimizer_G.zero_grad()
                        self.optimizer_E.zero_grad()
                        loss_manager.resetLosses()
                        loss_manager_D.resetLosses()
                        loss_manager_G.resetLosses()
                    else:
                        self.optimizer.zero_grad()
                        loss_manager.resetLosses()
                    

                    decoded_obs, decoded_next_obs = None, None
                    states_denoiser = None
                    states_denoiser_predicted = None
                    next_states_denoiser = None
                    next_states_denoiser_predicted = None

                    # Predict states given observations as in Time Contrastive Network (Triplet Loss) [Sermanet et al.]
                    if self.use_triplets:
                        states, positive_states, negative_states = self.model.forwardTriplets(obs[:, :3:, :, :],
                                                                                            obs[:, 3:6, :, :],
                                                                                            obs[:, 6:, :, :])

                        next_states, next_positive_states, next_negative_states = self.model.forwardTriplets(
                            next_obs[:, :3:, :, :],
                            next_obs[:, 3:6, :, :],
                            next_obs[:, 6:, :, :])
                    elif self.use_autoencoder:
                        (states, decoded_obs), (next_states, decoded_next_obs) = self.model(obs), self.model(next_obs)

                    elif self.use_dae:
                        (states, decoded_obs), (next_states, decoded_next_obs) = \
                            self.model(noisy_obs), self.model(next_noisy_obs)

                    elif self.use_vae:
                        (decoded_obs, mu, logvar), (next_decoded_obs, next_mu, next_logvar) = self.model(obs), \
                                                                                            self.model(next_obs)
                        states, next_states = self.model.getStates(obs), self.model.getStates(next_obs)

                        if self.perceptual_similarity_loss:
                            # Predictions for the perceptual similarity loss as in DARLA
                            # https://arxiv.org/pdf/1707.08475.pdf
                            (states_denoiser, decoded_obs_denoiser), (next_states_denoiser, decoded_next_obs_denoiser) = \
                                self.denoiser(obs), self.denoiser(next_obs)

                            (states_denoiser_predicted, decoded_obs_denoiser_predicted) = self.denoiser(decoded_obs)
                            (next_states_denoiser_predicted, decoded_next_obs_denoiser_predicted) = self.denoiser(next_decoded_obs)
                    else:
                        states, next_states = self.model(obs), self.model(next_obs)

                    # Actions associated to the observations of the current minibatch
                    
                    actions_st = action.view(-1, 1).to(self.device)
                    # L1 regularization
                    if self.losses_weights_dict['l1_reg'] > 0:
                        l1Loss(loss_manager.reg_params,
                            self.losses_weights_dict['l1_reg'], loss_manager)
                        l1Loss(loss_manager_D.reg_params,
                            self.losses_weights_dict['l1_reg'], loss_manager_D)
                        l1Loss(loss_manager_G.reg_params,
                            self.losses_weights_dict['l1_reg'], loss_manager_G)
                        

                    if self.losses_weights_dict['l2_reg'] > 0:
                        l2Loss(loss_manager.reg_params,
                            self.losses_weights_dict['l2_reg'], loss_manager)
                        l2Loss(loss_manager_D.reg_params,
                            self.losses_weights_dict['l2_reg'], loss_manager_D)
                        l2Loss(loss_manager_G.reg_params,
                            self.losses_weights_dict['l2_reg'], loss_manager_G)

                    if not self.no_priors:
                        roboticPriorsLoss(states, next_states, minibatch_idx=minibatch_idx,
                                        dissimilar_pairs=dissimilar_pairs, same_actions_pairs=same_actions_pairs,
                                        weight=self.losses_weights_dict['priors'], loss_manager=loss_manager)

                    if self.use_forward_loss:
                        next_states_pred = self.model.forwardModel(states, actions_st)
                        forwardModelLoss(next_states_pred, next_states,
                                        weight=self.losses_weights_dict['forward'],
                                        loss_manager=loss_manager)

                    if self.use_inverse_loss:
                        actions_pred = self.model.inverseModel(states, next_states)
                        inverseModelLoss(actions_pred, actions_st, weight=self.losses_weights_dict['inverse'],
                                        loss_manager=loss_manager)

                    if self.use_reward_loss:
                        # rewards_st = rewa[minibatch_idx]].copy() #[TODO]
                        rewards_st = reward.copy()
                        # Removing negative reward
                        rewards_st[rewards_st == -1] = 0
                        rewards_st = torch.from_numpy(rewards_st).to(self.device)
                        rewards_pred = self.model.rewardModel(states, next_states)
                        rewardModelLoss(rewards_pred, rewards_st.long(), weight=self.losses_weights_dict['reward'],
                                        loss_manager=loss_manager)

                    if self.use_autoencoder or self.use_dae:
                        loss_type = "dae" if self.use_dae else "autoencoder"
                        autoEncoderLoss(obs, decoded_obs, next_obs, decoded_next_obs,
                                        weight=self.losses_weights_dict[loss_type], loss_manager=loss_manager)

                    if self.use_vae:

                        kullbackLeiblerLoss(mu, next_mu, logvar, next_logvar, loss_manager=loss_manager, beta=self.beta)

                        if self.perceptual_similarity_loss:
                            perceptualSimilarityLoss(states_denoiser, states_denoiser_predicted, next_states_denoiser,
                                                    next_states_denoiser_predicted,
                                                    weight=self.losses_weights_dict['perceptual'],
                                                    loss_manager=loss_manager)
                        else:
                            generationLoss(decoded_obs, next_decoded_obs, obs, next_obs,
                                        weight=self.losses_weights_dict['vae'], loss_manager=loss_manager)

                    if self.reward_prior:
                        # rewards_st = rewar[minibatch_idx]]
                        rewards_st = reward
                        rewards_st = torch.from_numpy(rewards_st).float().view(-1, 1).to(self.device)
                        rewardPriorLoss(states, rewards_st, weight=self.losses_weights_dict['reward-prior'],
                                        loss_manager=loss_manager)

                    if self.episode_prior:
                        episodePriorLoss(minibatch_idx, minibatch_episodes, states, self.prior_discriminator,
                                        BALANCED_SAMPLING, weight=self.losses_weights_dict['episode-prior'],
                                        loss_manager=loss_manager)
                    if self.use_triplets:
                        tripletLoss(states, positive_states, negative_states, weight=self.losses_weights_dict['triplet'],
                                    loss_manager=loss_manager, alpha=0.2)

                    if self.model_type == 'gan':
                        label_valid = torch.ones((obs.size(0), 1)).to(self.device)
                        label_fake = torch.zeros((obs.size(0), 1)).to(self.device)

                        # === Train the Discriminator first ===
                        sample_state = torch.randn((obs.size(0), self.state_dim),
                                        requires_grad=False).to(self.device)
                        fake_img = self.model.generator(sample_state)
                        # fake_loss = 
                        ganNonSaturateLoss(self.model.discriminator(fake_img.detach()), label_fake, weight=1.0, loss_manager=loss_manager_D, name="ns_loss_D_fake")
                        # real_loss = 
                        ganNonSaturateLoss(self.model.discriminator(obs), label_valid, weight=1.0, loss_manager=loss_manager_D, name="ns_loss_D_real")
                        # d_loss = (real_loss + fake_loss) / 2
                        loss_manager_D.updateLossHistory()
                        d_loss = loss_manager_D.computeTotalLoss()
                        
                        if valid_mode:
                            pass
                        else:
                            d_loss.backward()
                            self.optimizer_D.step()
                        epoch_loss_D += d_loss.item()
                        epoch_batches_D += 1
                        
                        #############################
                        # === Train the Generator ===
                        sample_state = torch.randn((obs.size(0), self.state_dim),
                                        requires_grad=False).to(self.device)
                        fake_img = self.model.generator(sample_state)
                        fake_rating = self.model.discriminator(fake_img)
                        ganNonSaturateLoss(fake_rating, label_valid, weight=1.0, loss_manager=loss_manager_G, name="ns_loss_G")
                        loss_manager_G.updateLossHistory()
                        g_loss = loss_manager_G.computeTotalLoss()
                        
                        if valid_mode:
                            pass
                        else:
                            g_loss.backward()
                            self.optimizer_G.step()
                        epoch_loss_G += g_loss.item()
                        epoch_batches_G += 1
                        
                        if not valid_mode:
                            train_loss_D = epoch_loss_D / float(epoch_batches_D)
                            train_loss_G = epoch_loss_G / float(epoch_batches_G)
                        else:
                            val_loss_D = epoch_loss_D / float(epoch_batches_D)
                            val_loss_G = epoch_loss_G / float(epoch_batches_G)
                        ##############################
                        # === Train the Encoder and the other components (e.g. forward/inverse/reward model) ===
                        state_pred = self.model.model(obs)
                        reconstruct_obs = self.model.generator(state_pred)
                        state_pred_next = self.model.model(next_obs)
                        reconstruct_obs_next = self.model.generator(state_pred_next)
                        autoEncoderLoss(obs, reconstruct_obs, next_obs, reconstruct_obs_next, 10000.0, loss_manager)
                        ##############################

                    # Compute weighted average of losses of encoder part (including 'forward'/'inverse'/'reward' models)
                    loss_manager.updateLossHistory()
                    loss = loss_manager.computeTotalLoss()
                    
                    if valid_mode:
                        # Only forward pass in the validation mode.
                        # DO NOT waste time to backpropagate i.e. loss.backward() !
                        pass
                    else:
                        # Backpropagate loss and update ('optimizer.step()') weights.
                        loss.backward()
                        if self.model_type == 'gan':
                            self.optimizer_E.step()
                        else:
                            self.optimizer.step()
                    epoch_loss += loss.item()
                    epoch_batches += 1

                    if not valid_mode:
                        train_loss = epoch_loss / float(epoch_batches)
                    else:
                        val_loss = epoch_loss / float(epoch_batches)
                    
                    if monitor_mode == 'loss':
                        if iter_ind % ITER_FLAG == 0 or (iter_ind == n_batch_per_epoch-1):
                            if not valid_mode:
                                print("\rEpoch {:3}/{}, {:.2%}, train_loss: {:.4f} | (elapsed time: {:.2f}s)".format(
                                    epoch + 1, N_EPOCHS, (iter_ind+1)/n_batch_per_epoch, train_loss, time.time() - start_time), end="")
                            else:
                                print("\r-------(valid): {:.2%}, val_loss: {:.4f} | (elapsed time: {:.2f}s)".format(
                                    (iter_ind+1)/n_batch_per_epoch, val_loss, time.time() - start_time), end="")
                    elif monitor_mode == 'pbar':
                        pbar.update(1)
                if valid_mode:
                    torch.set_grad_enabled(self.prev_mode)
                if monitor_mode == 'loss':
                    print()
                elif monitor_mode == 'pbar':
                    pbar.close()
                    if valid_mode:
                        current_loss = val_loss
                    else:
                        current_loss = train_loss
                    print("Epoch {:3}/{}, {:.2%}, loss: {:.4f}".format(epoch + 1, N_EPOCHS, (iter_ind+1)/n_batch_per_epoch, current_loss))

            
            # Even if loss_history is modified by LossManager
            # we make it explicit
            def update_loss_history(loss_manager, train_loss, val_loss, epoch_batches, epoch):
                loss_history = loss_manager.loss_history
                loss_history['train_loss'].append(train_loss)
                loss_history['val_loss'].append(val_loss)
                for key in loss_history.keys():
                    if key in ['train_loss', 'val_loss']:
                        continue
                    loss_history[key][-1] /= epoch_batches
                    if epoch + 1 < N_EPOCHS:
                        loss_history[key].append(0)
                return loss_history
            loss_history = update_loss_history(loss_manager, train_loss, val_loss, epoch_batches, epoch)
            if self.model_type == 'gan':
                loss_history_D = update_loss_history(loss_manager_D, train_loss_D, val_loss_D, epoch_batches_D, epoch)
                loss_history_G = update_loss_history(loss_manager_G, train_loss_G, val_loss_G, epoch_batches_G, epoch)
            
            # Save best model
            if val_loss < best_error: ## [TODO]
                best_error = val_loss
                torch.save(self.model.state_dict(), best_model_path)

            if np.isnan(train_loss):
                printRed("NaN Loss, consider increasing NOISE_STD in the gaussian noise layer")
                sys.exit(NAN_ERROR)

            # Then we print the results for this epoch:
            if (epoch + 1) % EPOCH_FLAG == 0:
                if figdir is not None:
                    self.model.eval()
                    with torch.no_grad():
                        # Optionally plot the current state space
                        print("Predicting states for all the observations...")
                        plotRepresentation(self.predStatesWithDataLoader(dataloader_test), rewards,
                                        #    add_colorbar=epoch == 0,
                                           name="Learned State Representation (Training Data)",
                                           path=os.path.join(figdir_repr, "Epoch_{}.png".format(epoch+1))) ## [TODO]
                        if self.model_type == 'gan':
                            images = make_grid([obs[0], reconstruct_obs[0], next_obs[0], reconstruct_obs_next[0]], nrow=2)
                            plotImage(deNormalize(detachToNumpy(images)), mode='cv2', save2dir=figdir_recon, index=epoch+1)
                        if self.use_autoencoder or self.use_vae or self.use_dae:
                            # Plot Reconstructed Image
                            if obs[0].shape[0] == 3:  # RGB
                                images = make_grid([obs[0], decoded_obs[0], obs[1], decoded_obs[1]], nrow=2) # , normalize=True, range=(0,1)
                                plotImage(deNormalize(detachToNumpy(images)), mode='cv2', save2dir=figdir_recon, index=epoch+1)
                                if self.use_dae:
                                    raise NotImplementedError
                                    plotImage(deNormalize(detachToNumpy(noisy_obs[0])), "Noisy Input Image (Train)")
                                if self.perceptual_similarity_loss:
                                    raise NotImplementedError
                                    plotImage(deNormalize(detachToNumpy(decoded_obs_denoiser[0])),
                                              "Reconstructed Image DAE")
                                    plotImage(deNormalize(detachToNumpy(decoded_obs_denoiser_predicted[0])),
                                              "Reconstructed Image predicted DAE")
                                

                            elif obs[0].shape[0] % 3 == 0:  # Multi-RGB
                                raise NotImplementedError
                                for k in range(obs[0].shape[0] // 3):
                                    plotImage(deNormalize(detachToNumpy(obs[0][k * 3:(k + 1) * 3, :, :])),
                                              "Input Image {} (Train)".format(k + 1))
                                    if self.use_dae:
                                        plotImage(deNormalize(detachToNumpy(noisy_obs[0][k * 3:(k + 1) * 3, :, :])),
                                                  "Noisy Input Image (Train)".format(k + 1))
                                    if self.perceptual_similarity_loss:
                                        plotImage(deNormalize(
                                            detachToNumpy(decoded_obs_denoiser[0][k * 3:(k + 1) * 3, :, :])),
                                            "Reconstructed Image DAE")
                                        plotImage(deNormalize(
                                            detachToNumpy(decoded_obs_denoiser_predicted[0][k * 3:(k + 1) * 3, :, :])),
                                            "Reconstructed Image predicted DAE")
                                    plotImage(deNormalize(detachToNumpy(decoded_obs[0][k * 3:(k + 1) * 3, :, :])),
                                              "Reconstructed Image {}".format(k + 1))

        # if SAVE_PLOTS:
            # plt.close("Learned State Representation (Training Data)")
            # plt.close("all")

        # Load best model before predicting states
        self.model.load_state_dict(torch.load(best_model_path))

        print("Predicting states for all the observations...")
        # return predicted states for training observations
        self.model.eval()
        with torch.no_grad():
            pred_states = self.predStatesWithDataLoader(dataloader_test)
        pairs_loss_weight = [k for k in zip(loss_manager.names, loss_manager.weights)]
        if self.model_type == 'gan':
            pairs_loss_weight += [k for k in zip(loss_manager.names, loss_manager_D.weights)]
            pairs_loss_weight += [k for k in zip(loss_manager.names, loss_manager_G.weights)]
            ## [Warning: the following line requires python >= 3.5]
            loss_history = {**loss_history, **loss_history_D, **loss_history_G} 
        return loss_history, pred_states, pairs_loss_weight
