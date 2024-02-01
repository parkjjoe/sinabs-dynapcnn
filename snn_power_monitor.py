import samna
import samnagui
import time
import torch
import matplotlib.pyplot as plt
import matplotlib.font_manager as font_manager

from torch import nn
from multiprocessing import Process
from sinabs.from_torch import from_model
from sinabs.backend.dynapcnn import DynapcnnNetwork
assert samna.__version__ >= '0.21.8', f"samna version {samna.__version__} is too low for this experiment"

# 3. Monitor the dynamic power
#######################################################################################################
# Monitor the dynamic power of the devkit i.e. the power after we deploy an SNN on it.
# a GUI module of samna which helps to monitor the real-time power plot

# # 3.1 Create a simple SNN
# # create a one layer CNN
# input_shape = (1, 128, 128)
#
# cnn = nn.Sequential(nn.Conv2d(in_channels=1, out_channels=1, kernel_size=(1, 1), stride=(2, 2), padding=(0, 0), bias=False),
#                     nn.ReLU())
#
# # assign a handcraft weight to CNN
# cnn[0].weight.data = torch.ones_like(cnn[0].weight.data, dtype=torch.float32) * 0.05
# # cnn to snn
# snn = from_model(cnn, input_shape=input_shape, batch_size=1).spiking_model
import sinabs.layers as sl
from torch import nn
from sinabs.activation.surrogate_gradient_fn import PeriodicExponential

input_shape = (2, 34, 34)

ann = nn.Sequential(
            nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
            nn.ReLU(),
            nn.AvgPool2d(kernel_size=2, stride=2),

            nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2),
            nn.ReLU(),

            nn.Flatten(),
            nn.Linear(32 * 2 * 2, 10),
)

# # init the model weights
# for layer in ann.modules():
#     if isinstance(layer, (nn.Conv2d, nn.Linear)):
#         nn.init.xavier_normal_(layer.weight.data)

model_path = "/home/parkjoe/PycharmProjects/sinabs-dynapcnn/saved_models/tutorial_nmnist_conversion_ann_deeper0.pth"
ann.load_state_dict(torch.load(model_path, map_location="cpu"))
# snn_bptt.eval()
#
snn = from_model(ann, input_shape=input_shape, add_spiking_output=True, batch_size=1).spiking_model

# snn to DynapcnnNetwork
dynapcnn_net = DynapcnnNetwork(snn=snn, input_shape=(2, 34, 34), discretize=True, dvs_input=False)
# input_shape = (2, 34, 34)
#
# snn_bptt = nn.Sequential(
#             nn.Conv2d(in_channels=2, out_channels=8, kernel_size=3, padding=1),
#             sl.IAFSqueeze(spike_threshold=1.0, batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(in_channels=8, out_channels=16, kernel_size=3, padding=1),
#             sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(in_channels=16, out_channels=32, kernel_size=3, padding=1),
#             sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
#             nn.AvgPool2d(kernel_size=2, stride=2),
#
#             nn.Conv2d(in_channels=32, out_channels=32, kernel_size=3, padding=1, stride=2),
#             sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
#
#             nn.Flatten(),
#             nn.Linear(32 * 2 * 2, 10),
#             sl.IAFSqueeze(batch_size=4, min_v_mem=-1.0, surrogate_grad_fn=PeriodicExponential()),
# )
# #
# # snn = from_model(ann, input_shape=input_shape, batch_size=1).spiking_model
# #
# model_path = "/home/parkjoe/PycharmProjects/sinabs-dynapcnn/saved_models/
# snn_bptt.load_state_dict(torch.load(model_path))
# ann.eval()
#
#dynapcnn_net = DynapcnnNetwork(snn=snn_bptt, input_shape=input_shape, dvs_input=True)

# 3.2 Create a devkit config based on the SNN
dynapcnn_device_str = "speck2fdevkit:0"
devkit_cfg = dynapcnn_net.make_config(device=dynapcnn_device_str, monitor_layers=["dvs"])

# 3.3 Construct a samna graph for power monitoring & visualization
# get device name
devices = samna.device.get_all_devices()
device_names = [each.device_type_name for each in devices]
print(device_names)

# open devkit
devkit = samna.device.open_device(device_names[0])

# get the handle of the stop-watch of the devkit
stop_watch = devkit.get_stop_watch()

# reset the stop-watch of devkit
# stop_watch = devkit.get_stop_watch()
# stop_watch.set_enable_value(True)
# stop_watch.reset()
# time.sleep(0.01)

# get the handle of the power monitor of the devkit
power_monitor = devkit.get_power_monitor()

# create samna node for power reading
power_source_node = power_monitor.get_source_node()
power_buffer_node = samna.BasicSinkNode_unifirm_modules_events_measurement()

# init samna graph
samna_graph = samna.graph.EventFilterGraph()

'''
=====  construct branches in the graph to define how data flows  =====
'''
# branch #1: DVS data visualization on GUI
_, _, streamer = samna_graph.sequential([devkit.get_model_source_node(), "Speck2fDvsToVizConverter", "VizEventStreamer"])

# branch #2: Collect power data
samna_graph.sequential([power_source_node, power_buffer_node])

# branch #3: Real time power data visualization on GUI
samna_graph.sequential([power_source_node, "MeasurementToVizConverter", streamer])

# define tcp port for data visualization
streamer_endpoint = 'tcp://0.0.0.0:40000'
streamer.set_streamer_endpoint(streamer_endpoint)

# start the samna graph
samna_graph.start()

# 3.4 Setup the GUI visualizer of samna
# init samna node for tcp communication
samna_node = samna.init_samna()
sender_endpoint = samna_node.get_sender_endpoint()
receiver_endpoint = samna_node.get_receiver_endpoint()
# wait tcp connection build up, this is necessary to open remote node.
time.sleep(1.0)

# define a function that start the gui visualizer then we run it in the sub-process
def run_visualizer_process(receiver_endpoint, sender_endpoint, visualizer_id):

    samnagui.runVisualizer(0.6, 0.6, receiver_endpoint, sender_endpoint, visualizer_id)

    return
# init sub-process for GUI
visualizer_id = 3
gui_process = Process(target=run_visualizer_process, args=(receiver_endpoint, sender_endpoint, visualizer_id))

# start the GUI process
gui_process.start()

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
        print("Successfully start the GUI process! You should see a window pop up")
        break

# setup visualizer
visualizer = getattr(samna, name)


# set visualizer's receiver endpoint to streamer's sender endpoint
visualizer.receiver.set_receiver_endpoint(streamer_endpoint)
# connect the receiver output to splitter inside the visualizer
visualizer.receiver.add_destination(visualizer.splitter.get_input_channel())

# add DVS plots to gui
activity_plot_id = visualizer.plots.add_activity_plot(128, 128, "DVS Layer")
plot = visualizer.plot_0
plot.set_layout(0, 0, 0.5, 0.89)
visualizer.splitter.add_destination("dvs_event", visualizer.plots.get_plot_input(activity_plot_id))

# add real time power plots to gui
power_plot_id = visualizer.plots.add_power_measurement_plot("power consumption", 5, ["io", "ram", "logic", "vddd", "vdda"])
plot_name = "plot_" + str(power_plot_id)
plot = getattr(visualizer, plot_name)
plot.set_layout(0, 0.75, 1.0, 1.0)
plot.set_show_x_span(10)
plot.set_label_interval(2)
plot.set_max_y_rate(1.5)
plot.set_show_point_circle(False)
plot.set_default_y_max(1)
plot.set_y_label_name("power (mW)")  # set the label of y axis
visualizer.splitter.add_destination("measurement", visualizer.plots.get_plot_input(power_plot_id))

visualizer.plots.report()

print("Now you should see a change on the GUI window!")

# 3.5 Apply devkit config and launch the devkit
# apply devkit config
devkit.get_model().apply_configuration(devkit_cfg)
time.sleep(0.1)

print("Now you should see the input from the dvs sensor on the GUI window!" )

# 3.6 Start to monitor dynamic power
sample_rate = 100
power_monitor.start_auto_power_measurement(sample_rate)

print("Now you should see the real-time power plot shows on the GUI window!")

# stop the experiment
input("Press Enter to exit.")
gui_process.terminate()
gui_process.join()

samna_graph.stop()
samna.device.close_device(devkit)
