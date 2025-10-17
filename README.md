# PySESM

PySESM is a PyTorch-based Python library that implements SESM
(Sparse-Encoded Surrogate Model).  PySESM is designed for high-performance
surrogate modeling and function approximation. It excels at
representing complex, high-dimensional functions by implementing a
powerful 'divide and conquer' strategy. The core architecture
partitions the input space into manageable blocks, each handled by a
local model. A key innovation is its use of a globally shared,
learnable dictionary of basis functions (e.g., Gaussians) combined
with block-specific sparse codes. This approach allows the model to
learn a rich, shared representation of the function's features while
using sparse, localized codes to efficiently capture the specific
behavior in different regions of the input space, resulting in a
highly flexible and scalable framework for scientific computing and
machine learning tasks.

## Prepare environment

With `conda` or `micromamba`, create your working environment with

    > conda create -n "sesm" python=3.12
    
Install your PyTorch according to your hardware configuration.  For
instance, if you only have CPU:

    > pip3 install torch torchvision --index-url https://download.pytorch.org/whl/cpu
    
or if you have GPU with CUDA 12.8:

    > pip3 install torch torchvision

For PyTorch it is preferable to check https://pytorch.org for the
proper up-to-date install configuration.

## Dependencies and SESM installation

This requires at least Python 3.12 and the dependencies listed in
requirements.txt.  Besides the Python core libraries, PySESM relies on
PyTorch and numpy, although additional libraries are used in the
examples for visualization and dataset creation.

This, will install PySESM and its dependencies:

    > pip install -e . 
    
or 

    > pip install -e . --use-pep517
	
The experiments, examples and so on need additional libraries that
you can install with

    > pip install -e ".[dev]"
	
    
However, if you prefer to install the dependency packages with
`micromamba` or `conda`, then you can use:

    > conda install numpy matplotlib scipy scikit-learn pandas plotly

## Directory structure


* **bin**
  Some generic utilities.
* **examples**
  Basic use cases and not so basic ones
* **experiments**
  More elaborated experimentation setups to compare SESM with other surrogate
  strategies
* **pysesm** 
  The source code of the PySESM library itself
* **unit_tests**
  Unit tests for the constituent parts of the library
* pyproject.toml
  Project description
* requirements.txt (deprecated)
  Libraries used.
