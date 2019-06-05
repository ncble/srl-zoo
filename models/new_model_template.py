import torch
import torch.nn as nn

# torchsummary is optional, but really helpful for scalable model !! (see example below)
from torchsummary import summary 
try:
    # relative import
    from .base_models import BaseModelSRL
    from .base_trainer import BaseTrainer
except:
    from models.base_trainer import BaseTrainer


class NewModelName(BaseModelSRL):
    """
    BaseModelSRL: a subclass of nn.Module of Pytorch with 
        - getStates (equi to self.forward(x))

    NewModelName should act like an encoder in the sense that 'forward' method should take 
    an observation (image) as input and return encoded state representation.

    """
    def __init__(self, state_dim, img_shape, *args, **kwargs):
        super().__init__()
        self.state_dim = state_dim # state dimension
        self.img_shape = img_shape # input image shape, assert img_shape is "channel first" !!!
        ## some example
        self.dummy_model = nn.Sequential(
            nn.Linear(np.prod(self.img_shape), self.state_dim)
        )
        # Note: for Conv2D: the "same" padding has the following formula:
        # if dilation=1, default:
        #     padding = (kernel_size - stride) / 2

        self.dummy_model_cnn = nn.Sequential(
            nn.Conv2d(self.img_shape[0], 32, kernel_size=7, stride=3, padding=2, bias=True),
            nn.MaxPool2d(kernel_size=3, stride=2)
        )
        # Using summary, we can get the output shape of format [-1, channels, high, width] !
        out_shape = summary(self.dummy_model_cnn, img_shape, show=False)  # [-1, channels, high, width]
        ### Do what you want ....

        #######################

    def forward(self, x):
        # Define new model here
        return x


class NewModelTrainer(BaseTrainer):
    """
    Define the training mechanism of custom model. Build the model by the method "build_model".
    Define the training dynamic at batch level by the method "train_on_batch"

    BaseTrainer is a subclass of nn.Module of Pytorch ! This allows us to call trainer params 
    by self.model.parameters() (where 'model' is an instance of NewModelTrainer). 

    """

    def __init__(self, state_dim=2, img_shape=(3, 224, 224)):
        super().__init__()
        self.state_dim = state_dim
        self.img_shape = img_shape

    def build_model(self, model_type=None):
        ## [Warning] It's necessary to define the attribute "self.model"
        self.model = NewModelName(self.state_dim, self.img_shape)
        ## model_type is optional
        # if model_type == "a":
        #     self.model = 
        # elif model_type == "b":
        #     self.model = 
        # else:
        #     raise NotImplementedError
    def train_on_batch(self, X, optimizer, loss_manager, valid_mode=False, device=torch.device('cpu')):
        """
        :param X (could multiple argument: X1, X2, etc) batch of samples/labels
        :param optimizer: pytorch optimizer
        :param loss_manager: collect loss tensors and loss history
        :param valid_model (bool) validation mode (or training mode)
        return loss: (scalar)
        """
        ## Define the training mechanism here
        loss = 0
        return loss

    

if __name__=="__main__":
    print("Start")
