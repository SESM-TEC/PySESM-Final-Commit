# PySESM

SESM is an approach to data representation that utilizes a shared
parametric dictionary to create surrogate models for data
sub-blocks. This enables efficient knowledge transfer between similar
contexts, enhancing model flexibility and performance.

## Prepare environment

With `conda` or `micromamba`, create your working environment with

    > conda create -n "sesm" python=3.9
    
Install your PyTorch according to your hardware configuration.  For
instance, if you only have CPU:

    > conda install pytorch torchvision torchaudio cpuonly -c pytorch
    
or if you have GPU with CUDA 12.1:

    > conda install pytorch torchvision torchaudio pytorch-cuda=12.1 -c pytorch -c nvidia

For PyTorch is preferable to check https://pytorch.org for the
proper install configuration.

## Dependencies and SESM installation

This, will install PySESM and its dependencies:

    > pip install -e . 
    
or 

    > pip install -e . --use-pep517
    
However, if you prefer to install the dependency packages with
`micromamba` or `conda`, then you can use:

    > conda install numpy matplotlib scipy scikit-learn pandas plotly

