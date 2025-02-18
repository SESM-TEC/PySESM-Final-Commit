from pysesm.blocks import UniformPartitionManager
import torch
import logging
import pytest

def test_map_points():
    """Tests whether _find_block correctly assigns blocks to each point"""
