# pysesm/sparse_coding/_factory_registration.py

# Import the concrete classes
from .ISTALayer import ISTALayer

# Import the factory
from ..factories import SparseCodingFactory

# Perform registrations
SparseCodingFactory.register("classic_ista", ISTALayer)
