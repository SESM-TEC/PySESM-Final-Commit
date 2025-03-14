'''
Device management for SESM/SSESM project
Allows central configuration of compute devices (CPU/GPU) with flexible
assignment for different components
'''

import torch
import logging
from typing import Dict, Optional, Union, Any
from pysesm.enums.DeviceTargetEnum import DeviceTarget

class DeviceManager:
    """
    Central manager for device assignments across the SESM/SSESM framework.
    
    This class handles device assignments for different components of the model
    and provides methods to move tensors and modules to their assigned devices.
    """
    
    def __init__(self, 
                 logger: logging.Logger,
                 default_device: Optional[str] = None,  # Ahora es opcional
                 device_map: Optional[Dict[Union[DeviceTarget, str], str]] = None):
        """
        Initialize the device manager with device assignments.
        
        Args:
            default_device (str, optional): Default device to use when no specific assignment exists.
                                            Si no se proporciona, se usa "cpu".
            device_map (Dict, optional): Mapping from DeviceTarget to device string.
                                        Example: {DeviceTarget.ISTA_LAYER: "cuda:0", 
                                                  DeviceTarget.DICTIONARY_LAYER: "cpu"}
        """
        self.default_device = default_device if default_device is not None else "cpu"
        self.device_map = {DeviceTarget.GLOBAL: self.default_device}
        if device_map:
            self.update_device_map(device_map)
        self._validate_devices()
        logger.info(f"Initializing DeviceManager with device configuration: {self}")
        
    def _validate_devices(self):
        """Validate that all specified devices are available"""
        for component, device in self.device_map.items():
            if device.startswith("cuda") and not torch.cuda.is_available():
                raise RuntimeError(f"CUDA requested for {component} but CUDA is not available")
            
            if device.startswith("cuda:"):
                try:
                    device_idx = int(device.split(":")[-1])
                    if device_idx >= torch.cuda.device_count():
                        raise RuntimeError(f"CUDA device {device_idx} requested but only "
                                          f"{torch.cuda.device_count()} devices available")
                except ValueError:
                    raise RuntimeError(f"Invalid CUDA device specification: {device}")
    
    def get_device(self, component: Union[DeviceTarget, str]) -> str:
        """
        Get the assigned device for a component.
        
        Args:
            component: The component to get the device for.
            
        Returns:
            str: The device string for the component.
        """
        if isinstance(component, str):
            try:
                component = DeviceTarget(component)
            except ValueError:
                # If string doesn't match enum, use it as a custom component name
                pass
                
        return self.device_map.get(component, self.device_map.get(DeviceTarget.GLOBAL, self.default_device))
    
    
    def update_device_map(self, new_device_map: Dict[Union[DeviceTarget, str], str]):
        """
        Update the device map with new assignments.
        
        Args:
            new_device_map: New device assignments to add or update.
        """
        self.device_map.update(new_device_map)
        self._validate_devices()
        
    def __str__(self) -> str:
        """String representation showing current device assignments"""
        result = [f"DeviceManager (default: {self.default_device})"]
        for component, device in self.device_map.items():
            component_name = component.value if isinstance(component, DeviceTarget) else component
            result.append(f"  {component_name}: {device}")
        return "\n".join(result)