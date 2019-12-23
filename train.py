"""
Training of neural network
"""
from datetime import datetime

import torch
from torch import nn, optim
from tqdm import tqdm

from src import misc
from src.models import LeNet
from src.vis import VisdomVisualizer

VISU = VisdomVisualizer()
BATCH_SIZE = 128
LABELS = [1, 2]


def perform_train_epoch(model, trainloader, criterion, optimizer, log_freq=10, chck_freq=150):
    """
    This function performs a training epoch on a trainloader. E.g. it performs gradient descent
    on mini-batches until all the samples of the dataset have been seen by the network.
    """

    model.train()
    total_loss = 0
    total_correct = 0
    total = 0
    progress_bar_iter = tqdm(enumerate(trainloader, 1), desc="Epoch", leave=False)

    for idx, (inputs, labels) in progress_bar_iter:

        optimizer.zero_grad()
        outputs = model(inputs)
        _, prediction = torch.max(outputs.data, 1)
        total_correct += (prediction == labels).sum().item()
        total += prediction.shape[0]
        loss = criterion(outputs, labels)
        total_loss += loss.item()
        loss.backward()
        optimizer.step()

        if idx % log_freq == 0:

            train_loss = total_loss / idx
            VISU.push_train_loss(train_loss)
            train_accuracy = total_correct / total
            VISU.push_train_accuracy(train_accuracy)
            progress_bar_iter.set_description("Epoch. Loss: {}, Accuracy: {}"\
                .format(train_loss, train_accuracy))

        if idx % chck_freq == 0:

            torch.save(model.state_dict(), datetime.now().strftime(("checkpoints/%H:%M:%S.torch")))


def evaluate_model(model, testloader, criterion):
    """
    This function evaluate the test error on a complete dataset.
    """

    with torch.no_grad():

        total_loss = 0
        total_correct = 0
        total = 0

        for idx, (inputs, labels) in tqdm(enumerate(testloader), desc="Evaluation", leave=False):

            outputs = model(inputs)
            _, prediction = torch.max(outputs.data, 1)
            total_correct += (prediction == labels).sum().item()
            total += prediction.shape[0]
            loss = criterion(outputs, labels)
            total_loss += loss.item()

    VISU.push_test_accuracy(total_correct/total)
    VISU.push_test_loss(total_loss/idx)


def trace_classes(model, testloader):
    """
    This function display some images of every classes in different windows in visdom.
    """

    with torch.no_grad():

        inputs, _ = iter(testloader).next()
        outputs = model(inputs)
        _, prediction = torch.max(outputs.data, 1)
        max_index = testloader.dataset.targets.max().item()

        for i in range(max_index+1):

            images = inputs[prediction == i]
            label = LABELS[i]
            VISU.push_class_images(images, label)


def train(epochs, lr=0.001, momentum=0.9, weight_decay=1e-4):
    """
    Trains a lenet on mnist for the indicated number of epochs
    """

    model = LeNet(LABELS)
    criterion = nn.CrossEntropyLoss()
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum, weight_decay=weight_decay)

    trainloader, testloader = misc.load_mnist_data(kept_labels=LABELS, batch_size=BATCH_SIZE)

    for _ in tqdm(range(epochs), desc="Training"):

        perform_train_epoch(model, trainloader, criterion, optimizer)
        evaluate_model(model, testloader, criterion)
        trace_classes(model, testloader)

    torch.save(model.state_dict(), datetime.now().strftime(("checkpoints/final-%H:%M:%S.t7")))

    return model


if __name__ == "__main__":

    train(50)
