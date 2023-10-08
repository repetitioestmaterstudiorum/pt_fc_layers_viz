from setuptools import setup, find_packages

setup(
    name='pt_fc_layers_viz',
    version='0.1',
    packages=find_packages(),
    install_requires=[
        'graphviz',
        'torch',
        'IPython'
    ],
    author='repetitioestmaterstudiorum',
    author_email='44611591+repetitioestmaterstudiorum@users.noreply.github.com',
    description='A visualization tool for PyTorch fully connected layers.',
    url='https://github.com/repetitioestmaterstudiorum/pt_fc_layers_viz',
)