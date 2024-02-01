"""
SynOps/s Bandwidth Constrains On The Hardware

https://synsense.gitlab.io/sinabs-dynapcnn/faqs/tips_for_training.html
"""
#######################################################################################################
# What Is SynOps/s Bandwidth?
#######################################################################################################
# All the operations involved in the lifecycle of a spike arriving at a layer until it updates the neuron's membrane potential (and generates a spike if appropriate).
# "Logic" -> "Kernel Memory Read" -> "Logic" -> "Neuron Read" -> "Neuron Write" -> "Logic"

#  Each core (layer ) on the DynapCNN/Speck has an upper limit (bandwidth of SynOps/s) on the number of synaptic-operations it can execute per second.
#######################################################################################################
# How To Spot A Model Exceeding The Bandwidth?
#######################################################################################################
# If the bandwidth of the chip is exceeded:
#   1. A significant delay on the output is observed when manually writing pre-recorded input events to the devkit.
#   2. The DVS events visualized by samnagui show striped edge pattern, and a significant delay can be observed for real-time input DVS events.
#######################################################################################################
# How To Estimate The SynOps/s For A Model?
#######################################################################################################
# SNNAnalyzer class
from sinabs.synopcounter import SNNAnalyzer

# dt refers to the time interval (micro-second) of a single time step of your spike train tensor
analyzer = SNNAnalyzer(my_snn, dt=my_raster_data_dt)
output = my_snn(my_raster_data)  # forward pass
layer_stats = analyzer.get_layer_statistics()

for layer_name, _ in my_snn.named_modules():
    synops_per_sec = layer_stats["parameter"][layer_name]["synops/s"]
    print(f"SynOps/s of layer {layer_name} is: {synops_per_sec}")
#######################################################################################################
# How To Prevent From Exceeding SynOps/s Bandwidth?
#######################################################################################################
# 1. Use smaller Conv kernel.
# 2. Use fewer numbers of Conv kernel, i.e. less output channels.
# 3. Add "SynOps Loss" as a regularization item during training:
import torch
from sinabs.synopcounter import SNNAnalyzer

# use SNNAnalyzer to obtain SynOps/s
analyzer = SNNAnalyzer(my_snn, dt=my_raster_data_dt)
output = my_snn(my_raster_data)  # forward pass
layer_statistics = analyzer.get_layer_statistics()["parameter"]
synops_of_every_layer = [
    stats["synops/s"] for layer, stats in layer_statistics.items()
]

# suppose my_snn is a 4-layer SNN
# manually set the upper limit for each layer
synops_upper_limit = [1e5, 2e5, 2e5, 1e4]

# calculate synops loss
synops_loss = 0
for synops_per_layer, limit in zip(synops_of_every_layer, synops_upper_limit):
    # punish only if the synops_per_layer higher than the limit
    residual = (synops_per_layer - limit) / limit
    synops_loss += torch.nn.functional.relu(residual)

# 4. Switch on the "decimator" on your DynapCNN Core:
from sinabs.backend.dynapcnn import DynapcnnNetwork

# generate the samna configuration
dynapcnn = DynapcnnNetwork(snn=YOUR_SNN, input_shape=(1, 128, 128), dvs_input=False)
# suppose your snn is a 3-layer model
samna_cfg = dynapcnn.make_config(device="speck2fmodule", chip_layers_ordering=[0, 1, 2])

# turn on the decimator on Core #0
layer_idx = 0
samna_cfg.cnn_layers[layer_idx].output_decimator_enable = True
# 1 spike passed for every 4 output spikes
# the number of output events from Core#0 will be reduced to 1/4 of the original
samna_cfg.cnn_layers[layer_idx].output_decimator_interval = 0b001
