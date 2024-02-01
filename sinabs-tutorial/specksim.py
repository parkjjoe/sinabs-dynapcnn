"""
Specksim

https://synsense.gitlab.io/sinabs-dynapcnn/getting_started/specksim.html
"""
#######################################################################################################
# Introduction
#######################################################################################################
# a high performance Spiking-Convolutional Neural Network simulator which is written in C++ and bound to our backend library samna
# simulates the SNN completely event-based
# The network architecture that normally could not have been deployed to our hardware due to memory constrains can be tested in specksim.
# completely bound to Python
#######################################################################################################
# Setup
#######################################################################################################
# pip install sinabs-dynapcnn
# pip install samna
#######################################################################################################
# Supported architecture
#######################################################################################################
# only supports sequential models
# Each weight layer (torch.nn.Conv2d/Linear) should be followed by a spiking layer (sinabs.layers.IAF/IAFSqueeze).
# The output layer of the network has to be a spiking layer.
#######################################################################################################
# Supported layers
#######################################################################################################
# Parameter layers
# supports torch.nn.Conv2d/Linear
# Linear layer will be converted to a Conv2d layer keeping all its features.
# Biases are not supported.

# Pooling layers
# supports torch.nn.AvgPool2d, sinabs.layers.SumPool2d
# AvgPool2d layer will be converted to a SumPool2d layer.
# The weights of the following parameter layer will be scaled down based on the kernel size of the pooling layer.
# Stride is considered to be equal to the kernel size for simulating the on-chip behaviour.

# Spiking layers
# supports sinabs.layers.IAF/IAFSqueeze
# not supports LIF
#######################################################################################################
# Input/Output
#######################################################################################################
# Specksim expects events in the format of np.record arrays (x, y, t, p).
# The 4 keys are converted in np.uint32 format.
import numpy as np

event_format = np.dtype([("x", np.uint32), ("y", np.uint32), ("t", np.uint32), ("p", np.uint32)])
#######################################################################################################
# How-to-use
#######################################################################################################
# Import
import torch.nn as nn
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn.specksim import from_sequential

# Define an artificial neural network
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

# You can then load weights.

# Convert to SNN using sinabs
snn = from_model(ann, add_spiking_output=True, batch_size=1).spiking_model

# now convert this model to a SpecksimNetwork using the method from_sequential
#######################################################################################################
# Convert sequential SNNs to SpecksimNetwork
input_shape = (1, 28, 28) # MNIST shape
specksim_network_dynapcnn = from_sequential(snn, input_shape=input_shape)
# The input shape of the network has to be passed explicitly.

# Convert from sequential DynapcnnNetwork to SpecksimNetwork
# To do a more realistic, sequential SNN -> DynapcnnNetwork (quantized) -> SpecksimNetwork.
from sinabs.backend.dynapcnn import DynapcnnNetwork

input_shape = (1, 28, 28) # MNIST shape
dynapcnn_network = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=False, discretize=True)
# the dynapcnn_network weights are quantized as we passed discretize=True
specksim_network_dynapcnn = from_sequential(dynapcnn_network, input_shape=input_shape)

# Send events to the simulated SNN
x = 10 # in pixels
y = 10 # in pixels
t = 100_000 # typically in microseconds
p = 0
input_event = np.array([x,y,t,p], dtype=specksim_network.output_dtype)
output_event = specksim_network_(input_event)
print(output_event)

# Monitoring hidden layers
# Only the spiking layers are allowed to be monitored to use SpecksimNetwork.add_monitor(int) or SpecksimNetwork.add_monitor(List[int]).
# For monitoring the Nth spiking layer, you can pass N. The indices start from 0.
specksim_network.add_monitor(0)
output_event = specksim_network(input_event)

# Use SpecksimNetwork.read_monitors(List[int]) or SpecksimNetwork.read_all_monitors() to read from multiple monitors
intermadiate_layer_events: np.record = specksim_network.read_monitor(0)
print(intermediate_layer_events)

# Monitoring hidden layer states
# The state updates cannot be stored as there are too many changes to store in the memory.
# The states can be read at the end of each call with read_spiking_layer_states(int) method.
states: List[List[List[int]]] = specksim_network.read_spiking_layer_states(0)
print(states)

# Resetting states
# In benchmarking we typically reset the spiking layer states.
specksim_network.reset_states()
#######################################################################################################
# Drawbacks and possible questions
#######################################################################################################
# 1. No training
# 2. No biases (set False)
# 3. Breath-first vs Depth-first
#    HW with the DYNAP-CNN architecture are completely asynchronous.
#    Specksim processes events in breadth-first manner.
# 4. Timestamping
#    Event timestamps do not play any role in calculation in the chip itself.
#    The output events do have timestamps, which are handled by the development boards.
#    We assign the timestamp of the input event that led to its creation.
# 5. Real-time
#    It is not possible to reliably run SNN in real-time using specksim because of certain delays.
#######################################################################################################
# Try it yourself
#######################################################################################################
# an example of running a converted SNN trained on MNIST in specksim: examples/mnist/specksim_network.py
