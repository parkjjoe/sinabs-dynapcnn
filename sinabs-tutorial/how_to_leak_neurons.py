"""
How To Leak The Neurons On The Devkit

https://synsense.gitlab.io/sinabs-dynapcnn/getting_started/notebooks/leak_neuron.html

1. Make sure you have a bias term for the torch.nn.Module.
2. Setting the external slow-clock frequency.

Then the bias term will be added to the neuron's membrane potential at every clock cycle.
"""
#######################################################################################################
import torch
import time
import samna
from torch import nn
from sinabs.layers import IAFSqueeze
from sinabs.backend.dynapcnn import DynapcnnNetwork
from sinabs.backend.dynapcnn.io import calculate_neuron_address, neuron_address_to_cxy

# Define a SNN with bias term on the Convolution Layer

snn = nn.Sequential(
    nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), bias=True),
    IAFSqueeze(min_v_mem=-1.0, spike_threshold=1.0, batch_size=1),
)

input_shape = (1, 2, 2)
# Set artificial values for vmem and bias

weight_value = 1.0
bias_value = -0.1
vmem_value = 1.0
snn[0].weight.data = torch.ones_like(snn[0].weight.data) * weight_value
snn[0].bias.data = torch.ones_like(snn[0].bias.data) * bias_value

# init the v_mem as 1.0(which means as same as the threshold)

_ = snn(torch.zeros(1, *input_shape))
snn[1].v_mem.data = torch.ones_like(snn[1].v_mem.data) * vmem_value
# Create samna ReadNeuronValue for inpecting the value of the Vmem

input_events = []

# create ReadNeuronValue event as input
for x in range(input_shape[2]):
    for y in range(input_shape[1]):
        ev = samna.speck2f.event.ReadNeuronValue()
        ev.layer = 0
        # output feature map size is the same as the input shape
        ev.address = calculate_neuron_address(x=x, y=y, c=0, feature_map_size=input_shape)
        input_events.append(ev)

print(input_events)
#######################################################################################################
# Deploy the snn to speck devkit

dynapcnn = DynapcnnNetwork(snn=snn, discretize=True, dvs_input=False, input_shape=input_shape)
# don't forget to set the slow clock frequency!
# here we set the frequency to 1Hz, which mean the Vmem should decrease after every 1 second
dynapcnn.to(device="speck2fdevkit", slow_clk_frequency=1)

# Check if neuron states decrease along with time pass by

neuron_states = dict()

for iter_times in range(1, 21):
    # write input
    dynapcnn.samna_input_buffer.write(input_events)
    time.sleep(0.5)
    print(f'----After {0.5 * iter_times} seconds:----')
    # get outputs
    output_events = dynapcnn.samna_output_buffer.get_events()

    for out_ev in output_events:
        c, x, y = neuron_address_to_cxy(out_ev.address, feature_map_size=input_shape)
        pre_neuron_state = neuron_states.get((c, x, y), 127)
        neuron_states.update({(c, x, y): out_ev.neuron_state})
        print(f"c:{c}, x:{x}, y:{y}, vmem:{out_ev.neuron_state}")
