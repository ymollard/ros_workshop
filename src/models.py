"""
Simple LeNet neural network
"""
import torch.nn as nn
import torch.nn.functional as F
import torch
import numpy as np

try:
    from . import misc
    from . import vis
except ImportError:
    import misc
    import vis

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
        self.conv1 = nn.Conv2d(1, 6, 5)
        self.conv2 = nn.Conv2d(6, 16, 5)
        self.fc1   = nn.Linear(256, 120)
        self.fc2   = nn.Linear(120, 84)
        self.fc3   = nn.Linear(84, len(classes))


    def forward(self, x):
        """
        This method is called on a batch of data to perform a forward pass at training time.
        It links the different layers to form the complete function performed by the network.
        """

        ##################
        # YOUR CODE HERE #
        ##################
        out = F.relu(self.conv1(x))
        out = F.max_pool2d(out, 2)
        out = F.relu(self.conv2(out))
        out = F.max_pool2d(out, 2)
        out = out.view(out.size(0), -1)
        out = F.relu(self.fc1(out))
        out = F.relu(self.fc2(out))
        out = self.fc3(out)

        return out

    def infer(self, x):
        """
        This method is called to perform a forward pass on a a single image at inference time.
        It must return the most likely label.
        """

        ##################
        # YOUR CODE HERE #
        ##################
        outputs = self.forward(torch.from_numpy(x).view(1, 1, 28, 28).float())
        _, prediction = torch.max(outputs.data, 1)
        out = self.classes[prediction]

        return out


def batch_mean(batch):
    """
    This function takes 3d tensor containing a batch of data as input and computes its mean. Must return a float.
    """

    ##################
    # YOUR CODE HERE #
    ##################
    return batch.mean().tolist()


def batch_std(batch):
    """
    This function takes a 3d tensor containing a batch of data as input and computes its standard deviation. Must return a float.
    """

    ##################
    # YOUR CODE HERE #
    ##################
    return batch.std().tolist()
    

def compute_params_count(model):
    """
    This function computes the number of parameters of the input model.
    """

    ##################
    # YOUR CODE HERE # 
    ##################
    return sum([a.numel() for a in model.parameters()])

if __name__ == "__main__":

    print("1) Loading data:")
    loader, _ = misc.load_mnist_data(kept_labels=[1, 2], batch_size=128)
    images, labels = iter(loader).next()
    print("Batch shape: {}".format(images.shape))
    print("Labels shape: {}".format(labels.shape))
    assert isinstance(batch_mean(images), float)
    assert isinstance(batch_std(images), float)
    print("Batch mean: {}".format(batch_mean(images)))
    print("Batch std: {}".format(batch_std(images)))
    vis.preview_loader(images, labels, "Batch sample")
    vis.show_image(images[0, 0].numpy(), "Mnist sample")
    
    print("Ok\n\n2) Checking the model:")
    model = LeNet([1, 2])
    print(model)
    modules = list(model.modules())
    assert list(modules[1].parameters())[0].shape == torch.Size([6, 1, 5, 5])
    assert list(modules[2].parameters())[0].shape == torch.Size([16, 6, 5, 5])
    assert list(modules[3].parameters())[0].shape == torch.Size([120, 256])
    assert list(modules[4].parameters())[0].shape == torch.Size([84, 120])
    assert list(modules[5].parameters())[0].shape == torch.Size([2, 84])
    infer = model.infer(np.random.randn(28, 28))
    assert infer in [1, 2]
    output = model(images)
    _, prediction = torch.max(output.data, 1)
    vis.preview_loader(images, prediction, "Inference sample")
   
    print("Ok\n\n3) Computing the number of parameters")
    assert isinstance(compute_params_count(model), int)
    print("Number of parameters of the model: {}".format(compute_params_count(model)))
    
    print("Ok\n\n4) Previewing first layer kernels")
    vis.preview_kernels(list(model.parameters())[0], "Input kernels")