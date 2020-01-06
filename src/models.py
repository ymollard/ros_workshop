"""
Simple LeNet neural network
"""
import torch.nn as nn
import torch.nn.functional as F
import torch

class LeNet(nn.Module):

    def __init__(self, classes):
        """
        The initializer of the network. The layers used in the network must be
        declared here.
        """

        super(LeNet, self).__init__()
        self.classes = classes

        ##################
        # YOUR CODE HERE #
        ##################
        # self.conv1 = nn.Conv2d(1, 6, 5)
        # self.conv2 = nn.Conv2d(6, 16, 5)
        # self.fc1   = nn.Linear(256, 120)
        # self.fc2   = nn.Linear(120, 84)
        # self.fc3   = nn.Linear(84, 2)

    def forward(self, x):
        """
        This method is called to perform a forward pass at training time. It links the different
        layers to form the complete function performed by the network.
        """

        ##################
        # YOUR CODE HERE #
        ##################
        # out = F.relu(self.conv1(x))
        # out = F.max_pool2d(out, 2)
        # out = F.relu(self.conv2(out))
        # out = F.max_pool2d(out, 2)
        # out = out.view(out.size(0), -1)
        # out = F.relu(self.fc1(out))
        # out = F.relu(self.fc2(out))
        # out = self.fc3(out)

        return out

    def infer(self, x):
        """
        This method is called to perform a forward pass at inference time. 
        """

        ##################
        # YOUR CODE HERE #
        ##################
        # outputs = self.forward(torch.from_numpy(x).view(1, 1, 28, 28).float())
        # _, prediction = torch.max(outputs.data, 1)
        # out = self.classes[prediction]

        return out
