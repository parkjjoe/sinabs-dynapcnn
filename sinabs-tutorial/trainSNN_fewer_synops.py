"""
Training an SNN with fewer synops

https://sinabs.readthedocs.io/en/v1.2.10/how_tos/synops_loss_snn.html

Define a spiking model.
"""
#######################################################################################################
import torch
import torch.nn as nn
import sinabs
import sinabs.layers as sl

class SNN(nn.Sequential):
    def __init__(self, batch_size):
        super().__init__(
            sl.FlattenTime(),
            nn.Conv2d(1, 16, 5, bias=False),
            sl.IAFSqueeze(batch_size=batch_size),
            sl.SumPool2d(2),
            nn.Conv2d(16, 32, 5, bias=False),
            sl.IAFSqueeze(batch_size=batch_size),
            sl.SumPool2d(2),
            nn.Conv2d(32, 120, 4, bias=False),
            sl.IAFSqueeze(batch_size=batch_size),
            nn.Flatten(),
            nn.Linear(120, 10, bias=False),
            sl.IAFSqueeze(batch_size=batch_size),
            sl.UnflattenTime(batch_size=batch_size),
        )

batch_size = 5
snn = SNN(batch_size=batch_size)
print(snn)
#######################################################################################################
# The SNNAnalyzer class tracks different statistics for spiking (IAF/LIF) and parameter (Conv2d/Linear) layers.
# The number of synaptic operations is part of the parameter layers.
analyzer = sinabs.SNNAnalyzer(snn)
print(f"Synops before feeding input: {analyzer.get_model_statistics()['synops']}")

rand_input_spikes = (torch.ones((batch_size, 10, 1, 28, 28)) ).float()
y_hat = snn(rand_input_spikes)
print(f"Synops after feeding input: {analyzer.get_model_statistics()['synops']}")

layer_stats = analyzer.get_layer_statistics()

for layer_name in layer_stats.keys():
    print(f"Layer {layer_name} tracks statistics {layer_stats[layer_name].keys()}")
#######################################################################################################
# Once we can calculate the total number synops, we might want to choose a target synops number as part of our objective function.
# If we set the number to low, the network will fail to learn anything as there won't be any activity at all.
# We're going to only optimise for number of synaptic operations given a constant input.
# Set the target to twice the number of operations of the untrained network.
# Find out the target number of operations
target_synops = 2 * analyzer.get_model_statistics()['synops'].detach_()

optim = torch.optim.Adam(snn.parameters(), lr=1e-3)

n_synops = []
firing_rates = []
for epoch in range(100):
    sinabs.reset_states(snn)
    sinabs.zero_grad(snn)
    optim.zero_grad()

    snn(rand_input_spikes)

    model_stats = analyzer.get_model_statistics()
    synops = model_stats['synops']
    firing_rate = model_stats['firing_rate']

    n_synops.append(synops.detach().cpu().numpy())
    firing_rates.append(firing_rate.detach().cpu().numpy())

    synops_loss = (target_synops - synops).square() / target_synops.square()
    synops_loss.backward()
    optim.step()

import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(2, 1, sharex=True)

ax1.plot(n_synops, label="Synops during training")
ax1.axhline(y=target_synops.item(), color='black', label="Target synops")
ax1.set_ylabel("Synaptic ops\nper mini-batch")
ax1.legend()

ax2.plot(firing_rates)
ax2.set_ylabel("Average firing rate\nacross all neurons")
ax2.set_xlabel("Epoch")
#######################################################################################################
layer_stats = analyzer.get_layer_statistics()

for layer_name in ['2', '5', '8', '11']:
    print(f"Layer {layer_name} has {layer_stats['spiking'][layer_name]['n_neurons']} neurons.")

fig, axes = plt.subplots(4, 1, sharex=True, figsize=(6,6))

for axis, layer_name in zip(axes, ['2', '5', '8', '11']):
    axis.hist(layer_stats['spiking'][layer_name]['firing_rate_per_neuron'].detach().numpy().ravel(), bins=10)
    axis.set_ylabel(f"Layer {layer_name}")
axes[0].set_title("Histogram of firing rates")
axes[-1].set_xlabel("Spikes / neuron / time step");
