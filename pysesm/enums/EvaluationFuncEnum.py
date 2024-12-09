from enum import Enum


class EvaluationFuncEnum(Enum):
    TWOD_MULT = ("2d_mult",)
    BMM_MULT = ("bmm_mult",)
    DEFAULT = "default"
