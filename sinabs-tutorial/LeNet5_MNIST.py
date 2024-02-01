"""
sinabs Tutorial

Take LeNet as an example to train and test a spiking neural network (SNN).

https://sinabs.readthedocs.io/en/v1.2.10/tutorials/LeNet_5_EngChinese.html
"""
#######################################################################################################
# 1. Build/Define a LeNet CNN model in Pytorch
# 2. Train and test this LeNet CNN mode lin Pytorch
# 3. Convert this LeNet CNN model into SNN using sinabs
# 4. Test on SNN in sinbas
#######################################################################################################
import os
import torch
import torchvision
import sinabs
import sinabs.layers as sl
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from models.lenet5 import LeNet5
from tqdm.auto import tqdm
from torch.utils.data import DataLoader
from torchvision.datasets import MNIST
from sinabs.from_torch import from_model
#######################################################################################################
# Build/Define a LeNet CNN model in Pytorch
#######################################################################################################
# 1. Recommend to use torch.nn.Sequential of torch.nn layers instead of manually added forwarding functions among layers.
# 2. Current supporting standard layers:
#   Conv2d, Linear, AvgPool2d, MaxPool2d, ReLU, Flatten, Dropout, BatchNorm
# 3. Users can also define their own layers deriving from torch.nn.Module

# class LeNet5(nn.Module):
#     def __init__(self):
#         super(LeNet5, self).__init__()
#         self.seq = nn.Sequential(
#             # 1st Conv + ReLU + Pooling
#             nn.Conv2d(in_channels=1, out_channels=20, kernel_size=5, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             # 2nd Conv + ReLU + Pooling
#             nn.Conv2d(in_channels=20, out_channels=50, kernel_size=5, stride=1),
#             nn.ReLU(),
#             nn.MaxPool2d(kernel_size=2),
#             # Dense layers
#             nn.Flatten(),
#             nn.Linear(4 * 4* 50, 500),
#             nn.ReLU(),
#             nn.Linear(500, 10),
#         )
#
#     def forward(self, x):
#         return self.seq(x)

# Setting up environment
# 1. Torch device: GPU or CPU
# 2. Torch dataloader: training/testing/spiking_testing
# 3. Input image size: (n_channel, width, height)
def prepare():
    # Setting up environment

    # Declare global environment parameters
    global device, train_dataloader, test_dataloader, spiking_test_dataloader, input_image_size

    # Torch device
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")

    # Model folder to save trained models
    os.makedirs("LeNet5_MNIST_models", exist_ok=True)

    # Setting up random seed to reproduce experiments
    torch.manual_seed(0)
    if device != "cpu":
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False

    # Downloading/Loading MNIST dataset as tensors for training
    train_dataset = MNIST(
        "/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets",
        train=True,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )

    # Downloading/Loading MNIST dataset as tensors for testing
    test_dataset = MNIST(
        "/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets",
        train=False,
        download=False,
        transform=torchvision.transforms.ToTensor(),
    )

    # Define Torch dataloaders for training, testing and spiking testing
    BATCH_SIZE = 512
    train_dataloader = DataLoader(train_dataset, batch_size=BATCH_SIZE, shuffle=True)
    test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)
    spiking_test_dataloader = DataLoader(test_dataset, batch_size=BATCH_SIZE, shuffle=True)

    # Define the size of input images
    input_image_size = (1, 28, 28)

    # Return global parameters
    return (
        device,
        train_dataloader,
        test_dataloader,
        spiking_test_dataloader,
        input_image_size,
    )
#######################################################################################################
# Train LeNet CNN model in Pytorch
#######################################################################################################
# 1. Define Loss
# 2. Define optimizer
# 3. Backpropagation over batches and epochs
def train(model, n_epochs=20):
    # Training a CNN model

    # Define loss
    criterion = torch.nn.CrossEntropyLoss()
    # Define optimizer
    optimizer = torch.optim.Adam(model.parameters(), lr=1e-3)
    # Visualize and display training loss in a progress bar
    pbar = tqdm(range(n_epochs))

    # backprop over epochs
    for epoch in pbar:
        # over batches
        for imgs, labels in train_dataloader:
            # reset grad to zero for each batch
            optimizer.zero_grad()

            # port to device
            imgs, labels = imgs.to(device), labels.to(device)
            # forward pass
            outputs = model(imgs)
            # calculate loss
            loss = criterion(outputs, labels)
            # display loss in progres bar
            pbar.set_postfix(loss=loss.item())

            # backward pass
            loss.backward()
            # optimize parameters
            optimizer.step()
    return model
#######################################################################################################
# Test LeNet CNN model in Pytorch
#######################################################################################################
# Define the function to count the correct prediction
def count_correct(output, target):
    _, predicted = torch.max(output, 1)
    acc = (predicted == target).sum().float()
    return acc.cpu().numpy()

def test(model):
    # Test the accuracy of a CNN model

    # With no gradient means less memory and calculation on forward pass
    with torch.no_grad():
        # evaluation usese Dropout and BatchNorm in inference mode
        model.eval()
        # Count correct prediction and total test number
        n_correct = 0
        n_test = 0

        # over batches
        for imgs, labels in test_dataloader:
            # port to device
            imgs, labels = imgs.to(device), labels.to(device)
            # inference
            outputs = model(imgs)
            n_correct += count_correct(outputs, labels)
            n_test += len(labels)
    # calculate accuracy
    ann_accuracy = n_correct / n_test * 100.0
    print("ANN test accuracy: %.2f" % (ann_accuracy))
    return ann_accuracy
#######################################################################################################
# Test LeNet SNN model in sinabs
#######################################################################################################
# 1. Transfer pytorch trained CNN model to SNN model in sinabs
#   * neural model is different
#       - a spiking neuron of an SNN holds a membrane potential state (V) of a certain time t over a time period (n_dt)
#       - weighted input adds up to the V
#       - a spiking neuron outputs a spike (binary output per time step dt) when V >= threshold at time t
#       - once a spike is generated, the V is subtracted by membrane_subtract, and lower bound of V is set to min_v_mem
#   * network architecture is the same (e.g. convolution, pooling and dense)
#   * network parameters are the same (e.g. weights and biases)
# 2. Tile an image to a sequence of n_dt images as input to SNN simulations
#   * This processing on tile-up images seems inefficient
#   * however, it is only a software simulation on continuous current flow injecting to spiking neurons for n_dt length
#   * which is ultra power efficient on Neuromorphic hardware
# 3. sinabs can only infer one input as a time, so batch_size = 1
# 4. Classification is calculated on the count of spikes on the output layer over time period (n_dt)
# Define tensor_tile function to generate sequence of input images
def tensor_tile(a, dim, n_tile):
    # a: input tensor
    # dim: tile on a specific dim or dims in a tuple
    # n_tile: number of tile to repeat
    init_dim = a.size(dim)
    repeat_idx = [1] * a.dim()
    repeat_idx[dim] = n_tile
    a = a.repeat(*(repeat_idx))
    order_index = torch.LongTensor(
        np.concatenate([init_dim * np.arange(n_tile) + i for i in range(init_dim)])
    )
    return torch.index_select(a, dim, order_index)

def snn_test(model, n_dt=10, n_test=10000):
    # Testing the accuracy of SNN on sinabs
    # model: CNN model
    # n_dt: the time window of each simulation
    # n_test: number of test images in total

    # Transfer Pytorch trained CNN model to sinabs SNN model
    net = from_model(
        model, # Pytorch trained model
        input_image_size, # Input image size: (n_channel, width, height)
        spike_threshold=10, # Threshold of the membrane potential of a Spiking neuron
        bias_rescaling=1.0, # Subtract membrane potential when the neuron fires a spike
        min_v_mem=1.0, # The lower bound of the membrane potential
        num_timesteps=n_dt, # The number of time steps
    ).to(device)

    torch.save(net.state_dict(), os.path.join("LeNet5_MNIST_models", "snn_LeNet5.pth"))

    # With no gradient means less memory and calculation on forward pass
    with torch.no_grad():
        # evaluation usese Dropout and BatchNorm in inference mode
        net.spiking_model.eval()
        # Count correct prediction and total test number
        n_correct = 0
        # loop over the input files once a time
        for i, (imgs, labels) in enumerate(tqdm(spiking_test_dataloader)):
            if i > n_test:
                break
            # tile image to a sequence of n_dt length as input to SNN
            input_frames = tensor_tile(imgs, 0, n_dt).to(device)
            labels = labels.to(device)
            # reset neural states of all the neurons in the network for each inference
            net.reset_states()
            # inference
            outputs = net.spiking_model(input_frames)
            n_correct += count_correct(outputs.sum(0, keepdim=True), labels)
    # calculate accuracy
    snn_accuracy = n_correct / n_test * 100.0
    print("SNN test accuracy: %.2f" % (snn_accuracy))
    return snn_accuracy
#######################################################################################################
# Setting up environment
#######################################################################################################
prepare()
# Init LeNet5 CNN
classifier = LeNet5().to(device)
# Train CNN model
train(classifier, n_epochs=2)
# Test on CNN model
ann_accuracy = test(classifier)

# Test on SNN model
snn_accuracy = snn_test(classifier, n_dt=10, n_test=2000)

torch.save(classifier.state_dict(), os.path.join("LeNet5_MNIST_models", "ann_LeNet5.pth"))
