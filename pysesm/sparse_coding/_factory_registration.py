# pysesm/sparse_coding/_factory_registration.py

# Import the concrete classes
from .ISTALayer import ISTALayer
from .FISTALayer import FISTALayer
from .ADMMLayer import ADMMLayer

# Import the factory
from ..factories import SparseCodingFactory

# Perform registrations
SparseCodingFactory.register("classic_ista", ISTALayer)
SparseCodingFactory.register("fista", FISTALayer)
SparseCodingFactory.register("admm", ADMMLayer)
