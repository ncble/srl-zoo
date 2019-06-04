
import os, glob, sys
import numpy as np
from sklearn.decomposition import IncrementalPCA
import matplotlib.pyplot as plt
def center_reduce(A):
    moy = np.mean(A, axis=0)
    std = np.std(A, axis=0)
    return (A-moy), moy, std

if __name__=="__main__":
    print("Start")


    data_gt = np.load(
        '/home/lulin/Projects/4A_Intern/robotics-rl-srl/srl_zoo/data/mobile2D_fixed_tar_seed_0/ground_truth.npz')
    RobotPos = data_gt['ground_truth_states']
    TarPos = data_gt["target_positions"] ## shape = (num_episodes, 2) = (100, 2)
    TarPos = np.repeat(TarPos[:, None, :], 251, axis=1).reshape(-1, 2) # shape = (25100, 2)
    # X = np.load(
    #     "./mobile2D_fixed_tar_seed_0/19-06-03_21h53_57_custom_cnn_ST_DIM100_autoencoder/states_rewards.npz")["states"]
    X = np.load(
        "./mobile2D_fixed_tar_seed_0/19-06-03_22h31_06_custom_cnn_ST_DIM100_inverse_autoencoder/states_rewards.npz")["states"]
    
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
        
        






