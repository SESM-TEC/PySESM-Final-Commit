import setuptools

if __name__ == "__main__":
    setuptools.setup(
        name="sesm",
        version="0.0.1",
        install_requires=[
            'numpy>=2.0.0',
            'matplotlib>=3.9.1',
            'torch>=2.4.0',
            'scipy>=1.14.0',
            'logger>=3.12.4',
            'scikit-learn>=1.5.2'
        ],
        py_modules=[]
    )
