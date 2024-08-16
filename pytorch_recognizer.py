from torchvision import datasets
from torchvision.transforms import ToTensor
from torch.utils.data import DataLoader
import torch.nn as nn
import torch.nn.functional as F
import torch.optim as optim

# get train data
train_data = datasets.MNIST(
    root = 'data',
    train = True,
    transform = ToTensor(),
    download = True
)


# get test data
test_data = datasets.MNIST(
    root = 'data',
    train = False,
    transform = ToTensor(),
    download = True
)

# get the size of the train dataset
# print(train_data.targets.size())

# load the data to datches
loaders = {
    'train': DataLoader(train_data,         
                        batch_size=100, shuffle=True, num_workers=1),

    'test': DataLoader(train_data, 
                       batch_size=100, shuffle=True, num_workers=1),
}

# print(loaders)

# define the model architechture

class CNN_net(nn.Module):
    def __init__(self) -> None:
        super(CNN_net, self).__init__()

        self.conv1 = nn.Conv2d(1, 10, kernel_size=5)
        self.conv2 = nn.Conv2d(10, 2, kernel_size=5)
        self.conv2_drop = nn.Dropout2d()
        self.fc1 = nn.Linear(320, 50)
        self.fc2 = nn.Linear(50, 10)

    def forward(self, x):
        x = F.relu(F.max_pool2d(self.conv1(x), 2))
