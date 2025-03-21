from enum import Enum, auto

class HookType(Enum):
    """
    Enum representing the types of hooks available.
    """
    ISTALAYER = auto()  # Hook for ISTALayer
    DICTLAYER = auto()  # Hook for DictLayer