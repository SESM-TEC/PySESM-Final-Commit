from enum import Enum

class DeviceTarget(Enum):
    """Enum defining the various components that can have device assignments"""
    GLOBAL = "global"
    SPARSE_CODING_LAYER = "sparse_coding_layer"
    DICTIONARY_LAYER = "dictionary_layer"
    PARTITION_MANAGER = "partition_manager"
    #SURROGATE_FUNCTION = "surrogate_function"
