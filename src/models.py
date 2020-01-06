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

    def forward(self, x):
        """
        This method is called on a batch of data to perform a forward pass at training time.
        It links the different layers to form the complete function performed by the network.
        """

        ##################
        # YOUR CODE HERE #
        ##################

        return out

    def infer(self, x):
        """
        This method is called to perform a forward pass on a a single image at inference time.
        It must return the most likely label.
        """

        ##################
        # YOUR CODE HERE #
        ##################

        return out
