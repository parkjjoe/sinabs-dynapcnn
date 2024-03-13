import datetime
import os
import time
from collections import Counter

import samna
#######################################################################################################
import torch
from sinabs.backend.dynapcnn import DynapcnnNetwork
from tonic.datasets.nmnist import NMNIST
from torch.utils.data import Subset
from tqdm.notebook import tqdm

# Depoly SNN To The Devkit
#######################################################################################################
n_time_steps = 100

# cpu_snn = snn_convert.to(device="cpu")
root_dir = "/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets"

_ = NMNIST(save_to=root_dir, train=True)
_ = NMNIST(save_to=root_dir, train=False)

nmnist_train = NMNIST(save_to=root_dir, train=True)
nmnist_test = NMNIST(save_to=root_dir, train=False)

snn_convert = torch.load("/home/parkjoe/PycharmProjects/sinabs-dynapcnn/saved_models/tutorial_nmnist_conversion_deeper20240308_152441.pth")
print(snn_convert)

# cpu_snn = snn_convert.to(device="cpu")
cpu_snn = snn_convert.to(device="cpu")
dynapcnn = DynapcnnNetwork(snn=cpu_snn, input_shape=(2, 34, 34), discretize=True, dvs_input=True)
devkit_name = "speck2fdevkit"

# use the `to` method of DynapcnnNetwork to deploy the SNN to the devkit
dynapcnn.to(device=devkit_name, chip_layers_ordering="auto", monitor_layers=[-1])
print(f"The SNN is deployed on the core: {dynapcnn.chip_layers_ordering}")
#######################################################################################################
#devkit_cfg = dynapcnn.make_config(device=devkit_name, monitor_layers=["dvs"])
devices = samna.device.get_all_devices()
device_names = [each.device_type_name for each in devices]
print(device_names)
devkit = samna.device.open_device("Speck2fDevKit:0")

power_monitor = devkit.get_power_monitor()
power_source_node = power_monitor.get_source_node()
power_buffer_node = samna.BasicSinkNode_unifirm_modules_events_measurement()

samna_graph = samna.graph.EventFilterGraph()
samna_graph.sequential([power_source_node, power_buffer_node])
samna_graph.start()
power_monitor.start_auto_power_measurement(1) # 100 Hz sample rate
#######################################################################################################
# Inference On The Devkit
snn_test_dataset = NMNIST(save_to=root_dir, train=False)
# for time-saving, we only select a subset for on-chip infernce， here we select 1/100 for an example run
subset_indices = list(range(0, len(snn_test_dataset), 100))
#subset_indices = list(range(len(snn_test_dataset))) # all test data
snn_test_dataset = Subset(snn_test_dataset, subset_indices)

inference_p_bar = tqdm(snn_test_dataset)

test_samples = 0
correct_samples = 0
total_input_spikes = 0
total_output_spikes = 0

# Start to record inference time
start_time = time.time()

# for events, label in inference_p_bar:
for events, label in inference_p_bar:

    # create samna Spike events stream
    samna_event_stream = []
    for ev in events:
        spk = samna.speck2f.event.Spike()
        spk.x = ev['x']
        spk.y = ev['y']
        spk.timestamp = ev['t'] - events['t'][0]
        spk.feature = ev['p']
        # Spikes will be sent to layer/core #0, since the SNN is deployed on core: [0, 1, 2, 3]
        spk.layer = 0
        samna_event_stream.append(spk)

    # inference on chip
    # output_events is also a list of Spike, but each Spike.layer is 3, since layer#3 is the output layer
    output_events = dynapcnn(samna_event_stream)
    total_input_spikes += len(samna_event_stream)
    total_output_spikes += len(output_events)

    # use the most frequent output neruon index as the final prediction
    neuron_index = [each.feature for each in output_events]
    if len(neuron_index) != 0:
        frequent_counter = Counter(neuron_index)
        prediction = frequent_counter.most_common(1)[0][0]
    else:
        prediction = -1
    inference_p_bar.set_description(f"label: {label}, prediction: {prediction}， output spikes num: {len(output_events)}")

    if prediction == label:
        correct_samples += 1

    test_samples += 1

print(f"Total input spikes: {total_input_spikes}")
print(f"Total output spikes: {total_output_spikes}")
print(f"On chip inference accuracy: {correct_samples / test_samples:.4f}")

# Stop to record inference time
end_time = time.time()
# Calculate total inference time
total_inference_time = end_time - start_time
print(f"Total inference time on hareware: {total_inference_time} seconds")
#######################################################################################################
power_monitor.stop_auto_power_measurement()
samna_graph.stop()
power_events = power_buffer_node.get_events()

power_each_track = {}
event_count_each_track = {}
for evt in power_events:
    track_id = evt.channel
    power_value = evt.value
    power_each_track[track_id] = power_each_track.get(track_id, 0) + power_value
    event_count_each_track[track_id] = event_count_each_track.get(track_id, 0) + 1

print("Dynamic Power Measurements During Inference:")
for track_id in range(5):
    avg_power = (power_each_track[track_id] / event_count_each_track[track_id]) * 1000
    print(f"Track {track_id}: Average Power = {avg_power:.3f} mW")

samna.device.close_device(devkit)