# AGTensors
The project focuses on an implementation of automatic differentiation/gradients (AG) in python numpy. The goal was to design a flexible bulletproof implementaiton of AG that mimics the API of PyTorch. The project was designed purely for self study.

## ML module
The file [modules.py](src/modules.py) shows how one could use the AG implementation to implement modules/layers as in pytorch.
The [example](example/main.ipynb) builds a simple NN and uses it on a classification problem. 