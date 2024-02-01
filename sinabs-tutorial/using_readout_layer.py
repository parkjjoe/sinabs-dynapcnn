"""
Using Readout Layer

https://synsense.gitlab.io/sinabs-dynapcnn/getting_started/notebooks/using_readout_layer.html

speck2edevkit
"""
#######################################################################################################
import torch
import samna
import samnagui
import time
import random
import copy
import matplotlib.pyplot as plt

from torch import nn
from sinabs.backend.dynapcnn import DynapcnnNetwork
from multiprocessing import Process
from sinabs.from_torch import from_model
from sinabs.layers.pool2d import SumPool2d
from typing import Union
from matplotlib.ticker import MaxNLocator
#######################################################################################################
# 1. Create a 1-layer CNN which can be deployed to the devkit
#######################################################################################################
# init a cnn it has 2 out_channels for a binary classification task
# the input shape of this cnn is (1, 16, 16), output shape of this cnn is (2, 1, 1)

input_shape = (1, 16, 16)

cnn = nn.Sequential(SumPool2d(kernel_size=(1, 1)),
                    nn.Conv2d(in_channels=1, out_channels=2, kernel_size=(16, 16), stride=(1, 1), padding=(0, 0), bias=False),
                    nn.ReLU())

# assign the CNN layer with a handcraft weight:
#   the reasons:
#       1. create a bunch of fake input spikes X[x_0, x_1, ..., x_T] as the input of the devkit.
#       2. for any input spike x_t with timestamp t:
#           - if t, if t < 0.5 * T, the y coordinate of spike x_t will be in range [0, 7], i.e. on the top-half of the input region.
#           - if t, if t > 0.5 * T, the y coordinate of spike x_t will be in range [8, 15], i.e. on the bottom-half of the input region.
#       3. Based on this:
#           - the output spike from time 0 to 0.5 * T will all come from output channel #0.
#           - the output spike from time 0.5 * T to T will all come from output channel #1.

# set handcraft weights for the CNN
weight_ones = torch.ones(1, 8, 16, dtype=torch.float32)
weight_zeros = torch.zeros(1, 8, 16, dtype=torch.float32)

channel_1_weight = torch.cat([weight_ones, weight_zeros], dim=1).unsqueeze(0)
channel_2_weight = torch.cat([weight_zeros, weight_ones], dim=1).unsqueeze(0)
handcraft_weight = torch.cat([channel_1_weight, channel_2_weight], dim=0)

output_cnn_lyr_id = 1
cnn[output_cnn_lyr_id].weight.data = handcraft_weight
#######################################################################################################
# speck2edevkit neuron index remap
# If your devkit is not speck2e/speck2f, then you don't need this remap step.
def remapping_output_index(output_layer: Union[nn.Conv2d, nn.Linear]) -> Union[nn.Conv2d, nn.Linear]:
    """
    Since the mapping of output channel's index from last cnn layer to the readout layer is not correct
    for speck2e devkit.
    We need to remap the index for the channel's index.
    The mapping law is:
    readout_layer_channel <--> cnn_layer_output_channel
       1  <--> 0
       2  <--> 4
       x  <--> 4(x - 1)
          ...
       15 <--> 56
       Args:
           output_layer: Usually we can use both nn.Linear and nn.Conv2d as the output computational layer of
           a classifier. The shape of weights of those two different type layer are:
           nn.Linear -> [output_channel, input_channel]
           nn.Conv2d -> [output_channel, input_channel, *kernel_size]
       Returns:
           new_output_layer: mapped weight of last layer.
    """
    weights = output_layer.weight.data

    out_channel, input_channel, *rest_dims = weights.size()
    new_out_channel = (out_channel - 1) * 4 + 1

    new_weights = torch.zeros(new_out_channel, input_channel, *rest_dims, dtype=weights.dtype, device=weights.device)

    for channel_id in range(out_channel):
        new_weights[channel_id * 4, :] = weights[channel_id, :]

    output_layer.weight.data = new_weights

    # change the attributes for the parameter layer
    if isinstance(output_layer, nn.Conv2d):
        output_layer.out_channels = new_out_channel
    elif isinstance(output_layer, nn.Linear):
        output_layer.out_features = new_out_channel
    else:
        raise TypeError(
            f"Expect nn.Conv2d or nn.Linear but got {output_layer.__class__.__name__}"
        )

    return output_layer

# remapping the output layer's
cnn[output_cnn_lyr_id] = remapping_output_index(cnn[output_cnn_lyr_id])
#######################################################################################################
# cnn to snn
snn = from_model(cnn, input_shape=input_shape, batch_size=1).spiking_model
# snn to DynapcnnNetwork
dynapcnn_net = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=False)
#######################################################################################################
# 2. Two different types of output events from readout layer
#######################################################################################################
# ReadoutPinValue: https://synsense-sys-int.gitlab.io/samna/reference/speck2e/event/index.html#samna.speck2e.event.ReadoutPinValue
#                  It is generated only when the input spikes to one of the readout layer channel achieves the threshold
#                  of the readout layer during one slow-clock cycle. i.e. it is possible that ReadoutPinValue will not be
#                  generated at every slow-clock cycle.
#                  'index' attribute represents the neuron index with the max average input spikes.
# ReadoutValue: https://synsense-sys-int.gitlab.io/samna/reference/speck2e/event/index.html#samna.speck2e.event.ReadoutValue
#               It is generated at every slock-clock cycle.
#               'value' attribute is a number with 21 bits, but seeting up different output_model_sel, it could have different meaning.
'''
output_model_sel |   bit[20]  |         bit[19:16]          |            bit[15:0]
_________________________________________________________________________________________
       0b00      | data valid | neuron index of max average | power down (clock gating)
       0b01      | data valid | neuron index of max average | threshold compare output
       0b10      | data valid | neuron index of max average | average output of the selected neuron
       0b11      | data valid | neuron index of max average | average output of max spiking neuron
'''
# In this experiment, we use the mode '0b01'.
#######################################################################################################
# 3. Prepare for deployment
#######################################################################################################
# 3 parts need to set up to deploy the SNN to the devkit:
#   - devkit configuration: an instance of the samna.speckxx.configuration class
#   - samna graph: how the data flows into and out from the devkit
#   - visualizer (optional): a GUI visualizer to visualize:
#                           the input signal of the devkit, the output of the devkit, the real-time power consumption of the devkit

# 1. visualize the input of devkit
# 2. visualize data from readout layer
# 3. get data from last CNN layer

# 3.1 Create devkit configuration
readout_threshold = 1

# init devkit config
devkit_cfg = dynapcnn_net.make_config(device="speck2edevkit:0")

# ========== modify devkit config ==========

"""cnn layers configuration"""
# send to output spike from cnn output layer to readout layer as its input
cnn_output_layer = dynapcnn_net.chip_layers_ordering[-1]
# the readout layer id is fixed for speck2e devkit which is 12
readout_layer = 12
print(f'link output layer: {cnn_output_layer} to readout layer: {readout_layer}')
devkit_cfg.cnn_layers[cnn_output_layer].monitor_enable = True
devkit_cfg.cnn_layers[cnn_output_layer].destinations[0].enable = True
devkit_cfg.cnn_layers[cnn_output_layer].destinations[0].layer = readout_layer

"""readout layer configuration"""
devkit_cfg.readout.enable = True
devkit_cfg.readout.readout_configuration_sel = 0b11
devkit_cfg.readout.output_mode_sel = 0b01
devkit_cfg.readout.low_pass_filter_disable = True
devkit_cfg.readout.threshold = readout_threshold

"""dvs layer configuration"""
# link the dvs layer to the 1st layer of the cnn layers
devkit_cfg.dvs_layer.destinations[0].enable = True
devkit_cfg.dvs_layer.destinations[0].layer = dynapcnn_net.chip_layers_ordering[0]
# merge the polarity of input events
devkit_cfg.dvs_layer.merge = True
# drop the raw input events from the dvs sensor, since we write events to devkit manually
devkit_cfg.dvs_layer.pass_sensor_events = False
# enable monitoring the output from dvs pre-preprocessing layer
devkit_cfg.dvs_layer.monitor_enable = True

# 3.2 Construct samna graph
# open devkit
device_names = [each.device_type_name for each in samna.device.get_all_devices()]
print(f"Open device: {device_names[0]}")
devkit = samna.device.open_device(device_names[0])

# init the graph
samna_graph = samna.graph.EventFilterGraph()

# init necessary nodes in samna graph
# node for writing fake inputs into devkit
input_buffer_node = samna.BasicSourceNode_speck2e_event_speck2e_input_event()
# node for reading ReadoutValue
readout_value_buffer_node = samna.BasicSinkNode_speck2e_event_output_event()
# node for reading ReadoutPinValue
pin_value_buffer_node = samna.BasicSinkNode_speck2e_event_output_event()
# node for reading Spike(i.e. the output from last CNN layer)
spike_buffer_node = samna.BasicSinkNode_speck2e_event_output_event()

# build input branch for graph
samna_graph.sequential([input_buffer_node, devkit.get_model_sink_node()])

# build output branches for graph
# branch #1: for the dvs input visualization
_, _, streamer = samna_graph.sequential(
    [devkit.get_model_source_node(), "Speck2eDvsToVizConverter", "VizEventStreamer"])
# branch #2: for obtaining the ReadoutValue
_, type_filter_node_readout, _ = samna_graph.sequential(
    [devkit.get_model_source_node(), "Speck2eOutputEventTypeFilter", readout_value_buffer_node])
# branch #3: for obtaining the ReadoutPinValue
_, type_filter_node_pin, _ = samna_graph.sequential(
    [devkit.get_model_source_node(), "Speck2eOutputEventTypeFilter", pin_value_buffer_node])
# branch #4: for obtaining the output Spike from cnn output layer
_, type_filter_node_spike, _ = samna_graph.sequential(
    [devkit.get_model_source_node(), "Speck2eOutputEventTypeFilter", spike_buffer_node])

# set the streamer nodes of the graph
# tcp communication port for dvs input data visualization
streamer_endpoint = 'tcp://0.0.0.0:40000'
streamer.set_streamer_endpoint(streamer_endpoint)
# add desired type for filter node
type_filter_node_readout.set_desired_type("speck2e::event::ReadoutValue")
type_filter_node_pin.set_desired_type("speck2e::event::ReadoutPinValue")
type_filter_node_spike.set_desired_type("speck2e::event::Spike")

# start samna graph before using the devkit
samna_graph.start()

# 3.3 Set up visualizer
# init samna node for tcp transmission
samna_node = samna.init_samna()
sender_endpoint = samna_node.get_sender_endpoint()
receiver_endpoint = samna_node.get_receiver_endpoint()
visualizer_id = 3
time.sleep(1)  # wait tcp connection build up, this is necessary to open remote node.

# define a function that run the GUI visualizer in the sub-process
def run_visualizer(receiver_endpoint, sender_endpoint, visualizer_id):

    samnagui.runVisualizer(0.6, 0.6, receiver_endpoint, sender_endpoint, visualizer_id)

    return
# create the subprocess
gui_process = Process(target=run_visualizer, args=(receiver_endpoint, sender_endpoint, visualizer_id))
gui_process.start()
print("GUI process started, you should see a window pop up!")

# wait for open visualizer and connect to it.
timeout = 10
begin = time.time()
name = "visualizer" + str(visualizer_id)
while time.time() - begin < timeout:

    try:

        time.sleep(0.05)
        samna.open_remote_node(visualizer_id, name)

    except:

        continue

    else:

        visualizer = getattr(samna, name)
        print(f"successful connect the GUI visualizer!")
        break

# set up the visualizer and GUI layout

# set visualizer's receiver endpoint to streamer's sender endpoint for tcp communication
visualizer.receiver.set_receiver_endpoint(streamer_endpoint)
# connect the receiver output to splitter inside the visualizer
visualizer.receiver.add_destination(visualizer.splitter.get_input_channel())

# add plots to gui
activity_plot_id = visualizer.plots.add_activity_plot(128, 128, "DVS Layer")
plot = visualizer.plot_0
plot.set_layout(0, 0, 0.5, 0.89)

visualizer.splitter.add_destination("dvs_event", visualizer.plots.get_plot_input(activity_plot_id))
visualizer.plots.report()

print("now you should see a change on the GUI window!")

# 3.4 Set up readout layer's driven slow-clock
# The readout layer of the chip is driven by a slow-clock.
dk_io = devkit.get_io_module()
slow_clk_freq = 20 # Hz
dk_io.set_slow_clk_rate(slow_clk_freq)
dk_io.set_slow_clk(True)
#######################################################################################################
# 4. Start readout from devkit
#######################################################################################################
# 4.1 Create fake input for devkit
def create_fake_input_events(time_sec: int, data_rate: int = 1000):

    """
    Args:
        time_sec: how long is the input events
        data_rate: how many input events generated in 1 second

        During the first half time, it generates events where y coordinate only in range[0, 7] which means top half
        region of the input feature map.

        Then in the last half of time, it generates events where y coordinate only in range[8, 15] which means bottom
        half region of the input feature map.

    """

    time_offset_micro_sec = 5000  # make the timestamp start from 5000
    time_micro_sec = time_sec * 1000000  # timestamp unit is micro-second
    time_stride = 1000000 // data_rate

    half_time = time_micro_sec // 2

    events = []
    for time_stamp in range(time_offset_micro_sec, time_micro_sec + time_offset_micro_sec + 1, time_stride):

        spk = samna.speck2e.event.DvsEvent()
        spk.timestamp = time_stamp
        spk.p = random.randint(0, 1)
        spk.x = random.randint(0, 15)

        if time_stamp < half_time:
            spk.y = random.randint(0, 7)  # spike located in top half of the input region
        else:
            spk.y = random.randint(8, 15)  # spike located in bottom half of the input region

        events.append(spk)

    return events
# create fake input events
input_time_length = 3 # seconds
data_rate = 5000
input_events = create_fake_input_events(time_sec=3, data_rate=data_rate)

print(f"number of fake input spikes: {len(input_events)}")

# estimated slow-clock cycle for processing the input spikes
clock_cycles_esitmated = slow_clk_freq * input_time_length
print(clock_cycles_esitmated)

# 4.2 Read the ReadoutPinValue
# to read the ReadoutPinValue, we need to modify the devkit's readout layer's config a little bit
devkit_cfg.readout.monitor_enable = False
devkit_cfg.readout.readout_pin_monitor_enable = True


# then apply the config to devkit
devkit.get_model().apply_configuration(devkit_cfg)
time.sleep(0.1)
# write the fake input into the devkit

# enable & reset the stop-watch of devkit, this is mainly for the timestamp processing for the input&output events.
stop_watch = devkit.get_stop_watch()
stop_watch.set_enable_value(True)
stop_watch.reset()
time.sleep(0.01)

# clear output buffer
pin_value_buffer_node.get_events()
spike_buffer_node.get_events()

# write through the input buffer node
input_time_length = (input_events[-1].timestamp - input_events[0].timestamp) / 1e6
input_buffer_node.write(input_events)
# sleep till all input events is sent and processed
time.sleep(input_time_length + 0.02)

# read the ReadoutPinValue from related buffer node
pin_value_events = pin_value_buffer_node.get_events()

print("You should see the input events through the GUI window!")

# the number of the ReadoutPinValue should be very close to (or the same as) the estimated clock cycle.
print(f"The estimated clock cycle is {clock_cycles_esitmated}")
print(f"Number of ReadoutPinValue events: {len(pin_value_events)}")

# get the timestamp of the output event
pin_value_timestamp = [each.timestamp for each in pin_value_events]
# shift timestep starting from 0
start_t = pin_value_timestamp[0]
pin_value_timestamp = [each - start_t for each in pin_value_timestamp]

# get the index of the output neuron with maximum output
neuron_id = [each.index for each in pin_value_events]

# plot the output neuron index vs. time
fig, ax = plt.subplots()
ax.scatter(pin_value_timestamp, neuron_id)
ax.set(xlim=(0, 3e6),ylim=(0, 2.5))
ax.set_xlabel("time( micro sec)")
ax.set_ylabel("neuron index")
ax.set_title("ReadoutPinValue")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))  # make y-axis only show integer

# 4.3 Read the ReadoutValue
# to read the ReadoutValue, we need to modify the devkit's readout layer's config also

# you can¡¯t monitor any Spike events from CNN layers or the DVS sensor anymore by setting
# `readout.monitor_enable = True`
devkit_cfg.readout.monitor_enable = True
# disable ReadoutPinvalue reading
devkit_cfg.readout.readout_pin_monitor_enable = False


# then re-apply the config to devkit
devkit.get_model().apply_configuration(devkit_cfg)
time.sleep(0.1)
# write the fake input into the devkit again

# enable & reset the stop-watch of devkit, this is mainly for the timestamp processing for the input&output events.
stop_watch = devkit.get_stop_watch()
stop_watch.set_enable_value(True)
stop_watch.reset()
time.sleep(0.01)

# clear output buffer
readout_value_buffer_node.get_events()

# write through the input buffer node
input_time_length = (input_events[-1].timestamp - input_events[0].timestamp) / 1e6
input_buffer_node.write(input_events)
# sleep till all input events is sent and processed
time.sleep(input_time_length + 0.02)

# read the buffer immediately after the process waiting finished.
# since every slock-clock cycle will append a new ReadoutValue event into the buffer
readout_value_events = readout_value_buffer_node.get_events()

print("You will not see any input events through the GUI window!")

# the number of the ReadoutValue should be very close to (or the same as) the estimated clock cycle.
print(f"The estimated clock cycle is {clock_cycles_esitmated}")
print(f"Number of ReadoutValue events: {len(readout_value_events)}")

# the ReadoutValue event don't have a timestamp, but we know it is generated at every slow-clock cycle
time_each_clock_cycle = 1e6 / slow_clk_freq
readout_value_timestamp = [idx * time_each_clock_cycle for idx in range(len(readout_value_events))]

# shift timestep starting from 0
start_t = readout_value_timestamp[0]
readout_value_timestamp = [each - start_t for each in readout_value_timestamp]

# get the index of the output neuron with maximum output
neuron_id = [(each.value & 0x0F0000) >> 16  for each in readout_value_events]

# plot the output neuron index vs. time
fig, ax = plt.subplots()
ax.scatter(readout_value_timestamp, neuron_id)
ax.set(xlim=(0, 3e6),ylim=(0, 2.5))
ax.set_xlabel("time( micro sec)")
ax.set_ylabel("neuron index")
ax.set_title("ReadoutValue")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# 4.4 Check the output from DynapCNN Layer
# get the output events from last DynapCNN Layer
dynapcnn_layer_events = spike_buffer_node.get_events()

# eliminate the Spikes come from layer 13
# since the input DVSEvents will be converted into Spikes and output from layer 13
# so the Spikes from layer 13 is the input itself
dynapcnn_layer_events = [each for each in dynapcnn_layer_events if each.layer != 13]

print(f"number of output spikes from DynacpCNN Layer: {len(dynapcnn_layer_events)}")

# get the timestamp of the output event
spike_timestamp = [each.timestamp for each in dynapcnn_layer_events]
# shift timestep starting from 0
start_t = spike_timestamp[0]
spike_timestamp = [each - start_t for each in spike_timestamp]

# get the neuron index of each output spike
neuron_id = [each.feature  for each in dynapcnn_layer_events]


# plot the output neuron index vs. time
fig, ax = plt.subplots()
ax.scatter(spike_timestamp, neuron_id)
ax.set(xlim=(0, 3e6),ylim=(-0.5, 4.5))
ax.set_xlabel("time( micro sec)")
ax.set_ylabel("neuron index")
ax.set_title("Spike")
ax.yaxis.set_major_locator(MaxNLocator(integer=True))

# we should see the neuron index are 0 and 4 if using speck2e, because of the incorrect map relationship on speck2e devkit

# stop devkit when experiment finished.

gui_process.terminate()
gui_process.join()

samna_graph.stop()
samna.device.close_device(devkit)
