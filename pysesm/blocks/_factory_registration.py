# pysesm/blocks/_factory_registration.py

# Import the concrete classes
from .UniformPartitionManager import UniformPartitionManager
# Import the factory
from ..factories import BlockManagerFactory

# Perform registrations
BlockManagerFactory.register("uniform_partition_manager", UniformPartitionManager)
