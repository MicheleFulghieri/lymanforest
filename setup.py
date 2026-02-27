from setuptools import setup, find_packages

setup(
    name='lymanforest',
    version='0.1.0',
    author='Michele',
    description='Pipeline to process the Lyman-alpha forest from TNG simulation and test the spectrum of a qso',
    packages=find_packages(),  
    install_requires=[
        'numpy',
        'h5py',
        'scipy',
	'numba',
	'matplotlib',
	'tqdm'
    ],
)
