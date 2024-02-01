"""
Play With SPECK's DVS

https://synsense.gitlab.io/sinabs-dynapcnn/getting_started/notebooks/play_with_speck_dvs.html

The resolution of DVS is 128x128.

By modifying the hardware configuration object, the "DVS Layer/Pre-Processing Layer" on the hardware can optionally apply:
    Polarity Merging, Pooling, Cropping, Mirroring, Switching ON/OFF Polarity, Hot-Pixel Filtering, Output Destination Layer Selecting, etc.
    (Output Destination Layer Selecting: The DVS layer also 2 output destination layers.)
"""
#######################################################################################################
# Monitor DVS Events
#######################################################################################################
# There are 2 related attributions of the hardware configuration objecct:
# samna.speckxx.configuration.SpeckConfiguration.dvs_layer.raw_monitor_enable:
#   If True, users can monitor the raw events that generated by the DVS.
#   The type of the monitored events will be samna.speckxx.event.DvsEvent.
#   The raw DVS events means that they directly come from the DVS array and they will not be effected by any pre-processing like:
#       cropping, mirroring, filtering etc.
# samna.speckxx.configuration.SpeckConfiguration.dvs+layer.monitor_enable:
#   If True, users can monitor the pre-processed DVS events.
#   The type of the monitored events will be samna.speckxx.event.Spike and it must with an attribute .layer = 13,
#   i.e. all samna.speckxx.event.Spike that comes from the #13 layer are the output events from DVS layer.
#   Pre-processing operations like cropping, mirroring, filtering will effect those events.

# The DynapcnnVisualizer is monitoring the samna.speckxx.event.Spike.
#######################################################################################################
# Merge Polarity
#######################################################################################################
# The easiest ways of merging the 2 polarity into 1 is:
#   1. set the input_shape argument as input_shape=(1, xx, xx) when init the DynapcnnNetwork.
#   2. manually modify the hardware configuration smana.speckxx.configuration.SpeckConfiguration.dvs_layer.merge = True
#       before it is applied to the devkit.
from torch import nn
from sinabs.layers import IAFSqueeze
from sinabs.backend.dynapcnn.dynapcnn_visualizer import DynapcnnVisualizer
from sinabs.backend.dynapcnn import DynapcnnNetwork


# create a dummy snn for DynapcnnNetwork initialization
snn = nn.Sequential(
    nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0)
)

# init DynapcnnNetwork
input_shape = (1, 128, 128)
dynapcnn = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True)

# deploy to speck devkit, use a different name if you're using a different version of the devkit
devkit_name = "speck2fmodule"
dynapcnn.to(device=devkit_name, monitor_layers=["dvs", -1])

# Utilize the DynapcnnVisualizer to visualize the DVS events.
# Dvs Plot: shows the DVS events
# Spike Count Plot: shows the number of output spikes curve of layer #0
# Power Measurement Plot: shows the io, RAM, logic power consumption (doesn't include the power consumption of the DVS)

# Please do not close the visualizer when running this notebook. Once the visualizer is closed, try to resetart the notebook.

visualizer = DynapcnnVisualizer(
    window_scale=(4, 8),
    dvs_shape=(128, 128),
    add_power_monitor_plot=True,
    spike_collection_interval=1000,  # milii-second
)

visualizer.connect(dynapcnn_network=dynapcnn)
#######################################################################################################
# Not Merge Polarity
#######################################################################################################
# input_shape = (2, xx, xx)
snn = nn.Sequential(
    nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0)
)

# make a new hardware configuration
input_shape = (2, 128, 128)
devkit_cfg_bi_polarity = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True).make_config(device=devkit_name,
                                                                                                monitor_layers=["dvs", -1])

# update the configuration
dynapcnn.samna_device.get_model().apply_configuration(devkit_cfg_bi_polarity)
#######################################################################################################
# Pooling
#######################################################################################################
# The easiest ways of utilizing the pooling functionality of the DVS layer is:
#   1. add a nn.AvgPool2d or sinabs.layers.SumPool2d as the very first layer of the sequential model.
#   2. manually modify the hardware configuration samna.speckxx.configuration.SpeckConfiguration.dvs_layer.pooling.x and .y.
#       The default .x and .y is 1. The available numbers of the .x and .y is {1, 2, 4}
snn = nn.Sequential(
    # this pooling operation will be executed in the DVS Layer when deployed to the hardware
    nn.AvgPool2d(kernel_size=(2, 2)),
    # the input of the 1st DynapCNN Core will be (1, 64, 64)
    nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0)
)

# make a new hardware configuration
input_shape = (1, 128, 128)
devkit_cfg_pool = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True).make_config(device=devkit_name,
                                                                                                monitor_layers=["dvs", -1])

# update the configuration
dynapcnn.samna_device.get_model().apply_configuration(devkit_cfg_pool)

# Users should observe that the visualizer¡¯s ¡°Dvs Plot¡± window is resized by enabling pooling functionality.
#######################################################################################################
# Cropping
#######################################################################################################
# There are 2 ways to crop the DVS's input region:
#   1. modify the input_shape argument when init the DynapcnnNetwork.
#   2. modify the hardware configuration samna.speckxx.configuration.SpeckConfiguration.dvs_layer.origin and .cut.
#       .origin defines the coordinate of the top-left point of the DVS array.
#       .cut defines the coordinate of the bottom-right point of the DVS array.
# 1 ---------------------------------------------------------------------------------------------------
snn = nn.Sequential(
    nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0)
)

# make a new hardware configuration that crop the input region to 32 x 32
input_shape = (1, 32, 32)
devkit_cfg_crop = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True).make_config(device=devkit_name,
                                                                                                monitor_layers=["dvs", -1])

# only use the top-left 32 x 32 region of the DVS array
print(f"The top-left coordinate: {devkit_cfg_crop.dvs_layer.origin}")
print(f"The bottom-right coordinate: {devkit_cfg_crop.dvs_layer.cut}")

# update the configuration
dynapcnn.samna_device.get_model().apply_configuration(devkit_cfg_crop)

# Users should see the Dvs Plot only shows a small region after the code block above is executed.
# 2 ---------------------------------------------------------------------------------------------------
snn = nn.Sequential(
    nn.Conv2d(1, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0)
)

# make a new hardware configuration that crop the input region to 64 x 64
input_shape = (1, 64, 64)
devkit_cfg_crop = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True).make_config(device=devkit_name,
                                                                                                monitor_layers=["dvs", -1])

# use the central 64 x 64 region of the DVS array.
devkit_cfg_crop.dvs_layer.origin.x = 31
devkit_cfg_crop.dvs_layer.origin.y = 31
devkit_cfg_crop.dvs_layer.cut.x = 94
devkit_cfg_crop.dvs_layer.cut.y = 94

# update the configuration
dynapcnn.samna_device.get_model().apply_configuration(devkit_cfg_crop)

# The DynapcnnVisualizer won't move the cropped region to the center of the window.
# Try to change the .origin coordinate from (31, 31) to (0, 0) and .cut from (94, 94) to (63, 63) and see what is different.
#######################################################################################################
# Mirroring
#######################################################################################################
# Only set the mirroring configuration by modifying the samna.speckxx.configuration.SpeckConfiguration.dvs_layer.mirror.x or .y
# or samna.speckxx.configuration.SpeckConfiguration.dvs_layers.mirror_diagonal = True. (default = False)
# 1 ---------------------------------------------------------------------------------------------------
snn = nn.Sequential(
    nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0)
)

# make a new hardware configuration
input_shape = (2, 128, 128)
devkit_cfg_mirror_x = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True).make_config(device=devkit_name,
                                                                                                monitor_layers=["dvs", -1])
# mirror the DVS events along with x-axis
devkit_cfg_mirror_x.dvs_layer.mirror.x = True

# update the configuration
dynapcnn.samna_device.get_model().apply_configuration(devkit_cfg_mirror_x)
# 2 ---------------------------------------------------------------------------------------------------
snn = nn.Sequential(
    nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0)
)

# make a new hardware configuration
input_shape = (2, 128, 128)
devkit_cfg_mirror_diagonal = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True).make_config(device=devkit_name,
                                                                                                monitor_layers=["dvs", -1])
# mirror along with diagonal line
devkit_cfg_mirror_diagonal.dvs_layer.mirror_diagonal = True

# update the configuration
dynapcnn.samna_device.get_model().apply_configuration(devkit_cfg_mirror_diagonal)
#######################################################################################################
# Disable DVS Events From Entering The Processor
#######################################################################################################
# Users might not want the DVS events to be sent to the DynapCNN Core under some cases
# (e.g. running the hardware inference with pre-recorded events).
# Set the hardware configuration samna.speckxx.configuration.SpeckConfiguration.dvs_layer.pass_sensor_events = False
snn = nn.Sequential(
    nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0)
)

# make a new hardware configuration
input_shape = (2, 128, 128)
devkit_cfg_not_pass_dvs = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True).make_config(device=devkit_name,
                                                                                                monitor_layers=["dvs", -1])
# do not let the DVS events be sent to processor
devkit_cfg_not_pass_dvs.dvs_layer.pass_sensor_events = False

# update the configuration
dynapcnn.samna_device.get_model().apply_configuration(devkit_cfg_not_pass_dvs)

# The DynapcnnVisualizer is monitoring the samna.speckxx.event.Spike instead of the samna.speckxx.event.DvsEvent
# The Spike Count Plot will constantly show 0 spikes because there is no events sent to the #0 DynapCNN layer as its input events.
#######################################################################################################
# Output Destination Layer Select
#######################################################################################################
# By modifying the samna.speckxx.configuration.SpeckConfiguration.dvs_layer.destinations,
# users can choose which 2 layers to output pre-processed DVS events to.
snn = nn.Sequential(
    nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0)
)

# make a new hardware configuration
input_shape = (2, 128, 128)
devkit_cfg_output_layer_select = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True).make_config(
    device=devkit_name,
    monitor_layers=["dvs", -1])
for destination_index in [0, 1]:
    print(
        f"destination layer {destination_index} enable: {devkit_cfg_output_layer_select.dvs_layer.destinations[destination_index].enable}")
    print(
        f"destination layer {destination_index} to: {devkit_cfg_output_layer_select.dvs_layer.destinations[destination_index].layer}")

# not send the DVS events to layer #0
devkit_cfg_output_layer_select.dvs_layer.destinations[0].enable = False
devkit_cfg_output_layer_select.dvs_layer.destinations[0].layer = 0

# update the configuration
dynapcnn.samna_device.get_model().apply_configuration(devkit_cfg_output_layer_select)
#######################################################################################################
# Switching On/Off Polarity
#######################################################################################################
# The hardware configuration samna.speckxx.configuration.SpeckConfiguration.dvs_layer.on_channel and .off_channel control the 2 polarities of the DVS.
# (default = True). If False, users can switch off one or both 2 channels.
snn = nn.Sequential(
    nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0)
)

# make a new hardware configuration
input_shape = (2, 128, 128)
devkit_cfg_switch_off_p = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True).make_config(
    device=devkit_name,
    monitor_layers=["dvs", -1])

# switch off one channel
# try to set both channels to False to see what will happen!
devkit_cfg_switch_off_p.dvs_layer.on_channel = True
devkit_cfg_switch_off_p.dvs_layer.off_channel = False

# update the configuration
dynapcnn.samna_device.get_model().apply_configuration(devkit_cfg_switch_off_p)
#######################################################################################################
# Hot-Pixel Filtering
#######################################################################################################
# By modifying the samna.speckxx.configuration.SpeckConfiguration.dvs_filter,
# users can apply a hot-pixel filter on the raw DVS events.

# Users need to provide an externel-slow-clock for the DVS filter block if want to use it by calling the API of samna's Unifirm Module.
snn = nn.Sequential(
    nn.Conv2d(2, 1, kernel_size=(3, 3), stride=(2, 2), padding=(1, 1), bias=False),
    IAFSqueeze(batch_size=1, min_v_mem=-1.0)
)

# make a new hardware configuration
input_shape = (2, 128, 128)
devkit_cfg_filter = DynapcnnNetwork(snn=snn, input_shape=input_shape, dvs_input=True).make_config(device=devkit_name,
                                                                                                  monitor_layers=["dvs",
                                                                                                                  -1])

# set the filtering config
devkit_cfg_filter.dvs_filter.enable = True
devkit_cfg_filter.dvs_filter.filter_size.x = 3
devkit_cfg_filter.dvs_filter.filter_size.y = 3
devkit_cfg_filter.dvs_filter.hot_pixel_filter_enable = True
devkit_cfg_filter.dvs_filter.threshold = 5

# set up the Unifirm/IO module
devkit_io = devkit.get_io_module()

# update the configuration
dynapcnn.samna_device.get_model().apply_configuration(devkit_cfg_filter)
