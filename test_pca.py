
import os, glob, sys
import numpy as np
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
def center_reduce(A):
    moy = np.mean(A, axis=0)
    std = np.std(A, axis=0)
    return (A-moy), moy, std
def compute_GTC(state_pred, ground_truth):
    """
    :param state_pred (np.array) shape (N, state_dim)
    :param ground_truth (np.array) shape (N, dim), usually dim = 2
    return GTC (np.array): max of correlation coefficients, shape (dim, )
    """
    assert len(state_pred.shape) == len(ground_truth.shape) == 2, "Input should be 2D array"
    std_sp = np.std(state_pred, axis=0) # shape (state_dim, )
    std_gt = np.std(ground_truth, axis=0) # shape (dim, )
    mean_sp = np.mean(state_pred, axis=0)
    mean_gt = np.mean(ground_truth, axis=0)
    # scalar product
    A = (state_pred-mean_sp)[..., None] * (ground_truth-mean_gt)[:, None, :] 
    corr = np.mean(A, axis=0) # shape (state_dim, dim)
    corr = corr / (std_sp[:, None] * std_gt[None, :])
    return np.max(corr, axis=0)
if __name__=="__main__":
    print("Start")


    data_gt = np.load(
        '/home/lulin/Projects/robotics-rl-srl/srl_zoo/data/mobile2D_fixed_tar_seed_0/ground_truth.npz')
    RobotPos = data_gt['ground_truth_states']
    TarPos = data_gt["target_positions"] ## shape = (num_episodes, 2) = (100, 2)
    TarPos = np.repeat(TarPos[:, None, :], 251, axis=1).reshape(-1, 2) # shape = (25100, 2)
    # X = np.load(
    #     "./mobile2D_fixed_tar_seed_0/19-06-03_21h53_57_custom_cnn_ST_DIM100_autoencoder/states_rewards.npz")["states"]
    X = np.load(
        "./logs/mobile2D_fixed_tar_seed_0/19-06-03_15h42_48_custom_cnn_ST_DIM200_autoencoder/states_rewards.npz")["states"]
    out = compute_GTC(X, RobotPos)
    print(out)
    print(X.shape)
    

    # ipca = IncrementalPCA(n_components=2, batch_size=128)
    # ipca.fit(X)
    # A = ipca.transform(X)
    # plt.figure()
    # # plt.plot(A[:, 0], A[:, 1], "o")
    # RobotPos = RobotPos - np.mean(RobotPos, axis=0)
    # plt.scatter(A[:, 0], A[:, 1], s=5)
    # plt.scatter(RobotPos[:, 0], RobotPos[:, 1], s=5)
    # plt.show()

    A = X
    # import ipdb
    # ipdb.set_trace()
    

    A, moy, std = center_reduce(A)
    RobotPos, moy1, std1 = center_reduce(RobotPos)
    TarPos, moy2, std2 = center_reduce(TarPos)
    
    for i in range(2):
        a = np.mean(A*RobotPos[:, i][:, None], axis=0)/(std1[i]*std)
        # import ipdb; ipdb.set_trace()
        print(np.max(np.abs(a)))

    
    # for i in range(2):
    #     a = np.mean(A*TarPos[:, i][:, None], axis=0)/(std2[i]*std)
    #     print(np.max(np.abs(a)))
        
        






