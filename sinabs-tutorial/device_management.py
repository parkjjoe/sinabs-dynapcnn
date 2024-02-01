"""
Device Management

https://synsense.gitlab.io/sinabs-dynapcnn/faqs/device_management.html
"""
#######################################################################################################
# How Do I List All Connected Devices And Their IDs?
#######################################################################################################
# method 1
from sinabs.backend.dynapcnn.io import get_device_map
from typing import Dict

device_map: Dict[str, 'DeviceInfo'] = get_device_map()

print(device_map)
