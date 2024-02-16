"""
Dynapcnn Visualizer

https://synsense.gitlab.io/sinabs-dynapcnn/getting_started/visualizer.html
"""
#######################################################################################################
# Introduction
#######################################################################################################
# devkits: mainly used for benchmarking
# samna library: support processing of events and interpretation of the output in a streaming fashion
# samna API: designed for low-level communication with the chips
# samnagui package: do some visualization for testing models in real-time and real-life conditions
#######################################################################################################
# Available plots in samnagui (https://synsense-sys-int.gitlab.io/samna/reference/ui/index.html)
#######################################################################################################
# Activity Plot: visualizes the events produced by the on-chip sensor
# Line Plot: allows visualization of the events produced by the model running on the chip, displays power consumption measurements
# Image Plot: displays an image (ex. denoting a predicted output class)
#######################################################################################################
# Useful samna nodes for visualizing and Just-In-Time (JIT) compiled nodes
#######################################################################################################
# Node
#   Events that are related to DVS events from the sensor are:
#       - DvsEventDecimate: eliminates L out of M events (set_decimation_fraction (M: int, L: int))
#       - DvsEventRescale: rescales events 'x / width' and 'y / height' (set_rescaling_cofefficients (width_coeff: int, height_coeff: int))
#       - DvsToVizConverter: converts events from sensor to visualization events
#                           The ones that are connected to the chip output spikes:
#                               - SpikeCollectionNode: picks output spikes at intervals (ms) and makes events that can be used out of them
#                                                      (set_interval_mili_sec (interval: int))
#                               - SpikeCountNode: counts how many events from each output received among feature_count events and outputs a visualizer event
#                                                 (set_feature_count (feature_count: int))
#                               - MajorityReadoutNode: among the events produced by SpikeCollectionNode selects the most active output channel
#                                                      used alongside the ImagePlot
# https://synsense-sys-int.gitlab.io/samna/filters.html

# Just-In-Time compiled nodes
# These nodes are also available for any devboard under samna.graph.jit{nameOfNode}.
#######################################################################################################
# Dynapcnn Visualizer
#######################################################################################################
# Setup
import os
current_folder_path = str(os.path.join(os.getcwd()))
file_tokens = current_folder_path.split("/")[:-3]
params_path = os.path.join( os.path.join("/", *file_tokens), "/home/parkjoe/PycharmProjects/sinabs-dynapcnn/saved_models/cifar10dvs_conversion_ann_20240131_104147.pth")
#icons_folder_path = os.path.join( os.path.join("/", *file_tokens), "examples/icons/")

# Import requirements
import torch
import torch.nn as nn
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork

# Define model
# ann = nn.Sequential(
#     nn.Conv2d(2, 16, kernel_size=2, stride=2, bias=False),
#     nn.ReLU(),
#     # core 1
#     nn.Conv2d(16, 16, kernel_size=3, padding=1, bias=False),
#     nn.ReLU(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     # core 2
#     nn.Conv2d(16, 32, kernel_size=3, padding=1, bias=False),
#     nn.ReLU(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     # core 7
#     nn.Conv2d(32, 32, kernel_size=3, padding=1, bias=False),
#     nn.ReLU(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     # core 4
#     nn.Conv2d(32, 64, kernel_size=3, padding=1, bias=False),
#     nn.ReLU(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     # core 5
#     nn.Conv2d(64, 64, kernel_size=3, padding=1, bias=False),
#     nn.ReLU(),
#     nn.AvgPool2d(kernel_size=2, stride=2),
#     # core 6
#     nn.Dropout2d(0.5),
#     nn.Conv2d(64, 256, kernel_size=2, bias=False),
#     nn.ReLU(),
#     # core 3
#     nn.Dropout2d(0.5),
#     nn.Flatten(),
#     nn.Linear(256, 128, bias=False),
#     nn.ReLU(),
#     # core 8
#     nn.Linear(128, 11, bias=False),
# )

# Load model weights from the example folder
ann = torch.load(params_path)

# Convert to SNN
# dvs_input = True: the model can receive input from the on-board sensor or an external DVS sensor
# discretize = True: the model can be ported to the chip
sinabs_model = from_model(ann, add_spiking_output=True, batch_size=1)

input_shape = (2, 128, 128)
hardware_compatible_model = DynapcnnNetwork(sinabs_model.spiking_model.cpu(), dvs_input=True, discretize=True, input_shape=input_shape)

# Port the model to the chip
hardware_compatible_model.to(device="speck2fdevkit", monitor_layers=["dvs", -1], chip_layers_ordering="auto")

# Use DynapcnnVisualizer
# Refer to notion.
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer
visualizer = DynapcnnVisualizer(
    window_scale=(4, 8),
    dvs_shape=(128, 128),
    add_power_monitor_plot=True,
    add_readout_plot=False,
    spike_collection_interval=500,
    #readout_images=sorted([os.path.join(icons_folder_path, f) for f in os.listdir(icons_folder_path)])
)

# Finally connect your model to the visualizer
visualizer.connect(hardware_compatible_model)

# Try it yourself
# example script: /examples/visualizer/gesture_viz.py