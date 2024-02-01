"""
Available Network Architecture

https://synsense.gitlab.io/sinabs-dynapcnn/faqs/available_network_arch.html
"""
#######################################################################################################
# Can I achieve a "Residual Connection" like ResNet does?
#######################################################################################################
# Only manually change the samna.dynapcnn.configuration.CNNLayerDestination.layer to make residual short-cut.
#######################################################################################################
# What If I Really Want to Use "Residual Connection"!
#######################################################################################################
# Let's say you want an architecture like below:
from torch import nn
from sinabs.layers import IAFSqueeze

class ResidualBlock(nn.Module):

    def __init__(self):

        super(ResidualBlock, self).__init__()
        self.conv1 = nn.Conv2d(1, 2, kernel_size=(1, 1), bias=False)
        self.iaf1 = IAFSqueeze(batch_size=1, min_v_mem=-1.0)

        self.conv2 = nn.Conv2d(2, 2, kernel_size=(1, 1), bias=False)
        self.iaf2 = IAFSqueeze(batch_size=1, min_v_mem=-1.0)

        self.conv3 = nn.Conv2d(2, 4, kernel_size=(1, 1), bias=False)
        self.iaf3 = IAFSqueeze(batch_size=1, min_v_mem=-1.0)

    def forward(self, x):

        tmp = self.conv1(x)
        tmp = self.iaf1(tmp)
        out = self.conv2(tmp)
        out = self.iaf2(tmp)
        # residual connection
        out += tmp
        out = self.conv3(out)
        out = self.iaf3(out)

        return out

# sinabs-dynapcnn can only parse Sequential
# define a Sequential first
import samna
from sinabs.backend.dynapcnn import DynapcnnNetwork

SNN = nn.Sequential(

    nn.Conv2d(1, 2, kernel_size=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0),

    nn.Conv2d(2, 2, kernel_size=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0),

    nn.Conv2d(2, 4, kernel_size=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0),
)

# make samna configuration
dynapcnn = DynapcnnNetwork(snn=SNN, input_shape=(1, 16, 16), dvs_input=False)
samna_cfg = dynapcnn.make_config(device="speck2kdevkit")

# samna_cfg.cnn_layers[layer].destinations[0] stores each core's first destination layers configuration
# check the default layer ordering
for layer in [0, 1, 2]:
    print(f"Is layer {layer} output turned on: {samna_cfg.cnn_layers[layer].destinations[0].enable}")
    print(f"The destination layer of layer {layer} is layer {samna_cfg.cnn_layers[layer].destinations[0].layer}")

# manually modify the samna config
# since 1 DYNAP-CNN core can have 2 destination layer
# we need to enable the core#0's 2nd output destination and target it to core#2
# so we need to modify samna_cfg.cnn_layers[0].destinations[1]

samna_cfg.cnn_layers[0].destinations[1].enable = True
samna_cfg.cnn_layers[0].destinations[1].layer = 2

# by applying the modification above, we not only send the output of core#0 to core#1 but also to core#2.
# which means we achieve the residual block!

# finally we just need to apply the samna configuration to the devkit, we finish the deployment.
devkit = samna.device.open_device("Speck2fModuleDevKit")
devkit.get_model().apply_configuration(samna_cfg)
