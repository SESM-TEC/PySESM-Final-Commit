import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="sesm",
        version="0.0.1",
        install_requires=[
            'numpy<2',
            'matplotlib>=3.9.1',
            'torch',
            'scipy',
            'logger'
        ],
        py_modules=[]
    )
