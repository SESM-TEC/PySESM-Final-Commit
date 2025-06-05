# pysesm/dictionaries/_factory_registration.py

# Import the concrete classes
from .GaussianDictLayer import GaussianDictLayer

# Import the factory
from ..factories import DictFactory

# Perform registrations
DictFactory.register("gaussian", GaussianDictLayer)
