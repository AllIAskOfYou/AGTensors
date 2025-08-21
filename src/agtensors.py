from abc import ABC, abstractmethod
import numpy as np
from operations import *

# ----------------------------------------------------------------------------------------
# AUTOGRAD TENSOR
# ----------------------------------------------------------------------------------------
# define the functions to create tensors
def zeros(shape, dtype=np.float32):
    return AGTensor(np.zeros(shape, dtype=dtype))

def ones(shape, dtype=np.float32):
    return AGTensor(np.ones(shape, dtype=dtype))

def tensor(data, dtype=np.float32):
    return AGTensor(np.array(data, dtype=dtype))

def exp(x):
    return AGTensor(None, (x,), Exp())

def log(x):
    return AGTensor(None, (x,), Log())

def lsum(x):
    return AGTensor(None, (x,), LSum())

def concat(x, axis=-1):
    return AGTensor(None, (x,), Concat(axis))

def sigmoid(x):
    return AGTensor(None, (x,), Sigmoid())



# define the class for the autograd tensor
class AGTensor:
    def __init__(self, data=None, inputs=None, op=None, constant=False):
        if (data is None) and (op is not None) and all([inp.data is not None for inp in inputs]):
            data = op(*[inp.data for inp in inputs])
        self.data = data
        self.shape = data.shape if data is not None else None
        self.grad = np.zeros_like(data) if data is not None else None
        self.inputs = inputs
        self.op = op
        self.order = None
        self.constant = constant

    # only perform current operation
    def forward_step(self):
        if self.op is None:
            return
        self.data = self.op.forward(*[inp.data for inp in self.inputs])
    
    # only update the grad of the input tensors
    def backward_step(self):
        grads = self.op.backward(self.grad, *[inp.data for inp in self.inputs])
        for inp, grad in zip(self.inputs, grads):
            if not inp.constant:
                inp.grad += grad

    # perform the forward pass
    def forward(self):
        if self.order is None:
            self.order = AGTensor.topo_sort(self)
        for tensor in self.order[::-1]:
            tensor.forward_step()
        return self

    # perform the backward pass
    def backward(self, grad=None):
        if self.order is None:
            self.order = AGTensor.topo_sort(self)
        self.grad = np.ones_like(self.data) if grad is None else grad
        for tensor in self.order:
            tensor.backward_step()
            
    # topological sort of the graph
    @staticmethod
    def topo_sort(tensor):
        visited = set()
        marked = set()
        order = []
        def visit(tensor):
            if tensor in visited:
                return
            if tensor in marked:
                raise ValueError("Cycle detected")
            if tensor.op is not None:
                marked.add(tensor)
                for inp in tensor.inputs:
                    visit(inp)
                visited.add(tensor)
                order.insert(0, tensor)
        visit(tensor)
        return order
    
    # recursively update the gradients
    def zero_grad(self):
        self.grad = np.zeros_like(self.data)
        if self.op is not None:
            for inp in self.inputs:
                inp.zero_grad()

    # set the data of the tensor 
    def set_data(self, data):
        self.data[:] = data

    # operators overloading
    def __len__(self):
        return len(self.data)

    def __getitem__(self, index):
        return AGTensor(None, (self,), GetItem(index))
    
    def __setitem__(self, index, value):
        # insert another AGTensor into the graph before this AGTensor, that is a copy of this AGTensor
        tmp = AGTensor(self.data, self.inputs, self.op)
        if isinstance(value, AGTensor):
            self.inputs = (tmp, value)
            self.op = SetItem(index)
        else:
            self.inputs = (tmp,)
            self.op = SetItemConstant(index, value)
        data = None
        if all([inp.data is not None for inp in self.inputs]):
            data = self.op(*[inp.data for inp in self.inputs])
        self.data = data
    
    def __neg__(self):
        return AGTensor(None, (self,), Neg())

    def __add__(self, other):
        if isinstance(other, AGTensor):
            return AGTensor(None, (self, other), Add())
        return AGTensor(None, (self,), AddConstant(other))
    
    def __radd__(self, other):
        return AGTensor(None, (self,), AddConstant(other))
    
    def __sub__(self, other):
        if isinstance(other, AGTensor):
            return AGTensor(None, (self, other), Sub())
        return AGTensor(None, (self,), SubConstant(other))
    
    def __rsub__(self, other):
        return AGTensor(None, (self,), RSubConstant(other))
    
    def __mul__(self, other):
        if isinstance(other, AGTensor):
            return AGTensor(None, (self, other), Mul())
        return AGTensor(None, (self,), MulConstant(other))
    
    def __rmul__(self, other):
        return AGTensor(None, (self,), MulConstant(other))
    
    def __truediv__(self, other):
        if isinstance(other, AGTensor):
            return AGTensor(None, (self, other), Div())
        return AGTensor(None, (self,), DivConstant(other))
    
    def __rtruediv__(self, other):
        return AGTensor(None, (self,), RDivConstant(other))
    
    def __matmul__(self, other):
        return AGTensor(None, (self, other), MatMul())
    
    def __pow__(self, other):
        if isinstance(other, AGTensor):
            return AGTensor(None, (self, other), Power())
        return AGTensor(None, (self,), PowerConstant(other))
    
    def __rpow__(self, other):
        return AGTensor(None, (self,), RPowerConstant(other))
    
    def exp(self):
        return AGTensor(None, (self,), Exp())
    
    # other class methods    
    def sum(self, axis=None, keepdims=False):
        return AGTensor(None, (self,), Sum(axis, keepdims))
    
    def mean(self, axis=None, keepdims=False):
        return AGTensor(None, (self,), Mean(axis, keepdims))
    
    def t(self):
        return AGTensor(None, (self,), Transpose())
    
    def reshape(self, shape):
        return AGTensor(None, (self,), Reshape(shape))
    
    def flatten(self):
        return AGTensor(None, (self,), Reshape(-1))
    
    def astype(self, dtype):
        self.data = self.data.astype(dtype)
        return self
