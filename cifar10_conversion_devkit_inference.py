import torch
import samna
import time
from torchvision import datasets, transforms
from tqdm.notebook import tqdm
from sinabs.backend.dynapcnn import DynapcnnNetwork
from collections import Counter
from torch.utils.data import Subset
from torch.utils.data import DataLoader
#######################################################################################################
# Depoly SNN To The Devkit
#######################################################################################################
epochs = 5
lr = 1e-3
batch_size = 4
num_workers = 4
n_time_steps = 100

# cpu_snn = snn_convert.to(device="cpu")
transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))])

root_dir = "/home/parkjoe/PycharmProjects/sinabs-dynapcnn/datasets"
cifar10_test_dataset = datasets.CIFAR10(root=root_dir, train=False, download=True, transform=transform)
cnn_test_dataloader = DataLoader(cifar10_test_dataset, batch_size=batch_size, num_workers=num_workers, drop_last=True, shuffle=False)

snn_convert = torch.load("/home/parkjoe/PycharmProjects/sinabs-dynapcnn/saved_models/cifar10_conversion_20240205_095621.pth")
print(snn_convert)

cpu_snn = snn_convert.to(device="cpu")
dynapcnn = DynapcnnNetwork(snn=cpu_snn, input_shape=(3, 32, 32), discretize=True, dvs_input=False)
devkit_name = "speck2fdevkit"

# use the `to` method of DynapcnnNetwork to deploy the SNN to the devkit
dynapcnn.to(device=devkit_name, chip_layers_ordering="auto")
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
power_monitor.start_auto_power_measurement(100) # 100 Hz sample rate
#######################################################################################################
# Inference On The Devkit
# for time-saving, we only select a subset for on-chip infernce， here we select 1/100 for an example run
subset_indices = list(range(0, len(cifar10_test_dataset), 100)) # Use only 100 test images
#subset_indices = list(range(len(cifar10_test_dataset))) # Use all test images (10000)
snn_test_dataset = Subset(cifar10_test_dataset, subset_indices)

def cifar10_to_spike(data, n_time_steps=n_time_steps, max_rate=255):
    """
    Convert CIFAR-10 images to spike data

    Param:
        data: CIFAR-10 image tensor (B x C x H x W)
        n_time_steps: time steps of spike data
        max_rate: a firing frequency of max spike

    Return:
        torch.Tensor: spike data (B x T x C x H x W)
    """
    # Convert pixel intensity to spike firing frequency
    spike_rates = data.float() / max_rate
    spike_rates = spike_rates.unsqueeze(2).repeat(1, 1, n_time_steps, 1, 1)
    spike_rates = spike_rates.permute(0, 2, 1, 3, 4)

    # Generate random spike
    spikes = torch.rand_like(spike_rates) < spike_rates
    return spikes

inference_p_bar = tqdm(snn_test_dataset)

test_samples = 0
correct_samples = 0
total_output_spikes = 0

# Start to record inference time
start_time = time.time()

for data, label in inference_p_bar:

    spike_data = cifar10_to_spike(data, n_time_steps=n_time_steps)

    # if torch.sum(spikes) == 0:
    #     print("No spikes found in the data")
    #     continue

    # Convert spike tensor to list of spike events
    spike_events = []
    for t in range(spike_data.size(1)):
        for c in range(spike_data.size(2)):
            for y in range(spike_data.size(3)):
                for x in range(spike_data.size(4)):
                    if spike_data[0, t, c, y, x] > 0:
                        spike = samna.speck2f.event.Spike()
                        spike.x = x
                        spike.y = y
                        spike.timestamp = t
                        spike.feature = c
                        spike.layer = 0
                        spike_events.append(spike)

    # if len(spike_events) == 0:
    #     print("No spike events generated")
    #     continue

    # inference on chip
    # output_events is also a list of Spike, but each Spike.layer is 3, since layer#3 is the output layer
    output_events = dynapcnn(spike_events)
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

print(f"Total output spikes: {total_output_spikes}")
print(f"On chip inference accuracy: {correct_samples / test_samples}")

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