"""
Contains PyTorch model code to instantiate a TinyVGG and FCN model.
"""
import torch
from torch import nn

# Define the neural network
class MultiInputModel(nn.Module):
    def __init__(self):
        super(MultiInputModel, self).__init__()
        self.conv2d_1 = nn.Sequential(
            nn.Conv2d(in_channels=1, 
                      out_channels=16, 
                      kernel_size=3, # how big is the square that's going over the image?
                      stride=1, # default
                      padding=1),# options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number 
            nn.ReLU(),
            nn.Conv2d(in_channels=16, 
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.ReLU(),
            nn.MaxPool2d(kernel_size=2,
                         stride=2) # default stride value is same as kernel_size
        )
        self.conv2d_2 = nn.Sequential(
            nn.Conv2d(16, 32, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(32, 32, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )
        self.conv2d_3 = nn.Sequential(
            nn.Conv2d(32, 64, 3, padding=1),
            nn.ReLU(),
            nn.Conv2d(64, 64, 3, padding=1),
            nn.ReLU(),
            nn.MaxPool2d(2)
        )

        self.conv1d_1 = nn.Sequential(
            nn.Conv1d(in_channels=3,
                     out_channels=16,
                     kernel_size=3, # how big is the square that's going over the image?
                     stride=1, # default
                     padding=1), # options = "valid" (no padding) or "same" (output has same shape as input) or int for specific number
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.Conv1d(in_channels=16,
                      out_channels=16*2,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm1d(num_features=16*2),
            nn.ReLU(),
        )

        self.conv1d_2 = nn.Sequential(
            nn.Conv1d(in_channels=16*2,
                      out_channels=16,
                      kernel_size=3,
                      stride=1,
                      padding=1),
            nn.BatchNorm1d(num_features=16),
            nn.ReLU(),
            nn.AvgPool1d(kernel_size=2,
                         stride=2)
        )

        self.fc1 = nn.Linear(64*5*4, out_features=64)
        self.fc2 = nn.Linear(16*20, out_features=64)
        self.fc = nn.Linear(128, 6)
          
    def forward(self, x1, x2):
        # print(x1.shape)
        x1 = self.conv2d_1(x1)
        # print(x1.shape)
        x1 = self.conv2d_2(x1)
        x1 = self.conv2d_3(x1)
        # print(x1.shape)
        x1 = x1.view(x1.size(0), -1)  # Flatten hidden_units*10*8
        # print(x1.shape)
        x1 = self.fc1(x1)
        
        x2 = self.conv1d_1(x2)
        # print(x2.shape)
        x2 = self.conv1d_2(x2)
        # print(x2.shape)
        x2 = x2.view(x2.size(0), -1)  # Flatten hidden_units*20
        # print(x2.shape)
        x2 = self.fc2(x2)
        
        x = torch.cat((x1, x2), dim=1)  # Concatenate along feature dimension
        # print(x.shape)
        x = self.fc(x)
        # print(x.shape)
        return x
