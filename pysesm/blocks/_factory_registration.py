# pysesm/blocks/_factory_registration.py

# Import the concrete classes
from .UniformPartitionManager import UniformPartitionManager
from .AdaptativePartitionManager import AdaptativePartitionManager
# Import the factory
from ..factories import BlockManagerFactory

# Perform registrations
BlockManagerFactory.register("uniform_partition_manager", UniformPartitionManager)
BlockManagerFactory.register("adaptative_partition_manager", AdaptativePartitionManager)
