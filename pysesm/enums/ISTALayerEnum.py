from enum import Enum

class ISTALayerEnum(Enum):
    CLASSIC = "CLASSIC"  # La implementación actual
    FISTA = "FISTA"     # Fast ISTA (futura implementación)
    ADAPTIVE = "ADAPTIVE" # ISTA con parámetros adaptativos (futuro)