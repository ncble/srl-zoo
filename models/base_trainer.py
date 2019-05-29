import torch
import torch.nn as nn


class BaseTrainer(nn.Module):
    def __init__(self):
        super().__init__()
    def build_model(self):
        raise NotImplementedError
    def train_on_batch(self, X, optimizers, loss_managers, valid_mode=False, device=torch.device('cpu')):
        raise NotImplementedError
    

if __name__=="__main__":
    print("Start")
