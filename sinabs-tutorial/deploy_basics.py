"""
The Basics

https://synsense.gitlab.io/sinabs-dynapcnn/the_basics.html

Automates deploying a SNN on devices based on DYNAP-CNN technology and enables quick deployment and testing of your models on to the devkits.
"""
#######################################################################################################
# TLDR;
#######################################################################################################
import torch
import torch.nn as nn
from typing import List
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork

ann = nn.Sequential(
    nn.Conv2d(1, 20, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Conv2d(20, 32, 5, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Conv2d(32, 128, 3, 1, bias=False),
    nn.ReLU(),
    nn.AvgPool2d(2,2),
    nn.Flatten(),
    nn.Linear(128, 500, bias=False),
    nn.ReLU(),
    nn.Linear(500, 10, bias=False),
)

# Load your weights or train the model
ann.load_state_dict(torch.load("model_params.pt"), map_location="cpu")

# Convert your model to SNN
sinabs_model = from_model(ann, add_spiking_output=True)  # Your sinabs SNN model

# Convert your SNN to `DynapcnnNetwork`
hw_model = DynapcnnNetwork(
    sinabs_model.spiking_model,
    discretize=True,
    input_shape=(1, 28, 28)
)

# Deploy model to a dev-kit
hw_model.to(device="dynapcnndevkit:0")

# Send events to chip
events_in: List["Spike"] =  ... # Load your events
events_out = hw_model(events_in)

# Analyze your output events
...
#######################################################################################################
# Model conversion to DYNAP-CNN core structure
#######################################################################################################
# Each cores (layers) in DYNAP-CNN based chips comprises 3 functionalities:
# 2D Convolution -> IAF -> SumPooling
# Accordingly, the DynapcnnLayer class is a sequential model with 3 layers:
# conv_layer, spk_layer, pool_layer
# The network structure needs to be converted into a sequence of DynapcnnLayers.
#######################################################################################################
# Layer conversion
#######################################################################################################
# AvgPool2d -> SumPool2d, Linear -> Conv2d (1x1)
#######################################################################################################
# Parameter quantization
#######################################################################################################
# 'discretize = True' converted the model parameters from floating point to fixed point representation.
#######################################################################################################
# Device selection
#######################################################################################################
# To se all the recognized devices:
from sinabs.backend.dynapcnn import io
print(io.device_types)
#######################################################################################################
# List of devices currently recognized by samna
#######################################################################################################
# A map of all device types and their corresponding samna `device_name`
device_types = {
    "speck": "speck",
    "dynapse2": "DYNAP-SE2 DevBoard",
    "dynapse2_stack": "DYNAP-SE2 Stack",
    "speck2devkit": "Speck2DevKit",
    "dynapse1devkit": "Dynapse1DevKit",
    "davis346": "Davis 346",
    "davis240": "Davis 240",
    "dvxplorer": "DVXplorer",
    "pollendevkit": "PollenDevKit",
    "dynapcnndevkit": "DynapcnnDevKit",
}
# or
from sinabs.backend.dynapcnn import io
io.get_all_samna_devices()

from sinabs.backend.dynapcnn.chip_factory import ChipFactory
ChipFactory.supported_devices
#######################################################################################################
# Placement of layers on device cores
#######################################################################################################
# For debugging, run ConfigBuilder.get_valid_mapping() and the object ConfigBuilder.get_constraints().
#######################################################################################################
# Porting model to device
#######################################################################################################
# Similar to portina a model to cpu with model.to("cpu") and GPU with model.to("cuda:0"),
# you can also port your DynapcnnCompatibleModel to a chip with model.to("dynapcnndevkit:0").
hw_model.to(
    device="dynapcnndevkit:0",
    chip_layers_ordering="auto",  # default value is "auto"
    monitor_layers=[-1], # the last layer of the model
    config_modifier=config_modifier, # for advanced users
)
#######################################################################################################
# Sending and receiving spikes
#######################################################################################################
# You can send a pre-defined sequence of events to the chip
events_out = hw_model(events_in)
########################################################################################################
# Monitoring layer activity
#######################################################################################################
# Use monitor_layers
# samna_output_buffer accumulates all the events sent out by the chip, including those from the monitored layers.