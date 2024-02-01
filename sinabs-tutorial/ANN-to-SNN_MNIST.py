"""
Converting an ANN to an SNN

https://sinabs.readthedocs.io/en/v1.2.10/tutorials/weight_transfer_mnist.html

"""
import os
#######################################################################################################
# Defining an ANN
#######################################################################################################
import torch.nn as nn

ann = nn.Sequential(
    nn.Conv2d(1, 20, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),

    nn.Conv2d(20, 32, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),

    nn.Conv2d(32, 128, 3, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2, 2),

    nn.Flatten(),
    nn.Linear(128, 500, bias=False),
    nn.ReLU(),
    nn.Linear(500, 10, bias=False),
)
#######################################################################################################
# Define a custom dataset
#######################################################################################################
# Override MNIST to also optionally return a spike raster instead of an image.
# Use rate coding to generate a series of spikes
import torch
from torchvision import datasets, transforms

class MNIST(datasets.MNIST):
    def __init__(self, root, train=True, is_spiking=False, time_window=100):
        super().__init__(
            root=root, train=train, download=True, transform=transforms.ToTensor()
        )
        self.is_spiking = is_spiking
        self.time_window = time_window

    def __getitem__(self, index):
        img, target = self.data[index].unsqueeze(0) / 255, self.targets[index]
        # img is now a tensor of 1x28x28

        if self.is_spiking:
            img = (torch.rand(self.time_window, *img.shape) < img).float()

        return img, target

# Fine-tune the ANN
# Training for standard image classification
from torch.utils.data import DataLoader

mnist_train = MNIST("/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets", train=True, is_spiking=False)
train_loader = DataLoader(mnist_train, batch_size=128, shuffle=True)

mnist_test = MNIST("/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets", train=False, is_spiking=False)
test_loader = DataLoader(mnist_test, batch_size=128, shuffle=False)

from tqdm.auto import tqdm
import torch
import torch.nn.functional as F

device = "cuda:0" if torch.cuda.is_available() else "cpu"

ann = ann.to(device)
ann.train()

optim = torch.optim.Adam(ann.parameters(), lr=1e-3)

n_epochs = 2

for n in tqdm(range(n_epochs)):
    for data, target in iter(train_loader):
        data, target = data.to(device), target.to(device)
        output = ann(data)
        optim.zero_grad()

        loss = F.cross_entropy(output, target)
        loss.backward()
        optim.step()

correct_predictions = []

for data, target in iter(test_loader):
    data, target = data.to(device), target.to(device)
    output = ann(data)

    # get the index of the max log-probability
    pred = output.argmax(dim=1, keepdim=True)

    # compute the total correct predictions
    correct_predictions.append(pred.eq(target.view_as(pred)))

correct_predictions = torch.cat(correct_predictions)
print(f"Classification accuracy: {correct_predictions.sum().item()/(len(correct_predictions))*100}%")
#######################################################################################################
# Model conversion to SNN
# how to build an equivalent spiking convolutional neural network (SCNN)
# from_model method in sinabs that converts a standard CNN into a spiking neural network

from sinabs.from_torch import from_model

input_shape = (1, 28, 28)
num_timesteps = 100 # per sample

sinabs_model = from_model(ann, input_shape=input_shape, add_spiking_output=True, synops=False, num_timesteps=num_timesteps)
# input_shape is needed in order to instantiate a SNN with the appropriate number of neurons because SNNs are stateful.
# add_spiking_output is a boolean flag to specify whether or not to add a spiking layer as the last layer in the network.
# synops=True tells sinabs to include the machinery for calculating synaptic operations, which we'll use later.

# ReLU layers are replaced by SpikingLayer.
print(sinabs_model.spiking_model)

# Model validation in sinabs simulation
test_batch_size = 10

spike_mnist_test = MNIST("/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets", train=False, is_spiking=True, time_window=num_timesteps)
spike_test_loader = DataLoader(spike_mnist_test, batch_size=test_batch_size, shuffle=False)

import sinabs.layers as sl

correct_predictions = []

for data, target in tqdm(spike_test_loader):
    data, target = data.to(device), target.to(device)
    data = sl.FlattenTime()(data)
    with torch.no_grad():
        output = sinabs_model(data)
        output = output.unflatten(0, (test_batch_size, output.shape[0] // test_batch_size))

        # get the index of the max log-probability
        pred = output.sum(1).argmax(dim=1, keepdim=True)

        # compute the total correct predictions
        correct_predictions.append(pred.eq(target.view_as(pred)))
        if len(correct_predictions) * test_batch_size >= 300:
            break

correct_predictions = torch.cat(correct_predictions)
print(f"Classification accuracy: {correct_predictions.sum().item()/(len(correct_predictions))*100}%")
# A free parameter that was added time_window determines whether or not your SNN is going to work well.
# The longer time_window is, the more spikes we produce as input and the better the performance of the network is going to be.

# Visualisation of specific example

# get one sample from the dataloader
img, label = spike_mnist_test[10]

import matplotlib.pyplot as plt

# %matplotlib inline

plt.imshow(img.sum(0)[0]);
plt.figure()

snn_output = sinabs_model(img.to(device))

import numpy as np

plt.pcolormesh(snn_output.T.detach().cpu())

plt.ylabel("Neuron ID")
plt.yticks(np.arange(10) + 0.5, np.arange(10))
plt.xlabel("Time");
plt.show()
