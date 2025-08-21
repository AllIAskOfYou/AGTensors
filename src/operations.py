import numpy as np
from abc import ABC, abstractmethod
import agtensors as agt

# define the function to reverse broadcast the gradient
def revBroadcast(grad, shape):
    grad_shape = grad.shape
    if shape != grad_shape:
        if len(grad_shape) != len(shape):
            raise ValueError("Not implemented yet")
        axis = []
        for i in range(len(shape)):
            if shape[i] != grad_shape[i]:
                axis.append(i)
        return np.sum(grad, axis=tuple(axis), keepdims=True)
    return grad

# ------------------------------------------------------------------------------
# OPERATIONS
# ------------------------------------------------------------------------------

# define an abstract class for the operations
class Operation(ABC):
    def forward(self, *inputs):
        return self.compute(*inputs)

    def backward(self, output_grad, *inputs):
        grad = self.gradient(output_grad, *inputs)
        if not isinstance(grad, tuple):
            grad = (grad,)
        return grad
    
    @abstractmethod
    def compute(self, *inputs):
        raise NotImplementedError
    
    @abstractmethod
    def gradient(self, *inputs):
        raise NotImplementedError
    
    def __call__(self, *inputs):
        return self.forward(*inputs)
    
# define an abstract class for broadcasted operations
class BCOperation(Operation):
    def backward(self, output_grad, *inputs):
        grads = self.gradient(output_grad, *inputs)
        return tuple(
            [revBroadcast(grad, inp.shape) for grad, inp in zip(grads, inputs)])
    
# define the class for the getitem operation
class GetItem(Operation):
    def __init__(self, index):
        if isinstance(index, agt.AGTensor):
            index = index.data.squeeze()
        if isinstance(index, tuple):
            new_index = []
            for idx in index:
                if isinstance(idx, agt.AGTensor):
                    new_index.append(idx.data.flatten())
                else:
                    new_index.append(idx)
            index = tuple(new_index)
        self.index = index
    
    def compute(self, x):
        return x[self.index]
    
    def gradient(self, output_grad, x):
        grad = np.zeros_like(x)
        grad[self.index] = output_grad
        return grad

# define the class for the setitem operation
class SetItem(Operation):
    def __init__(self, index):
        self.index = index
    
    def compute(self, x, value):
        x = x.copy()
        x[self.index] = value
        return x
    
    def gradient(self, output_grad, x, value):
        x_grad = output_grad.copy()
        x_grad[self.index] = 0
        value_grad = output_grad[self.index]
        return x_grad, value_grad

# define the class for the constant setitem operation
class SetItemConstant(Operation):
    def __init__(self, index, constant):
        self.index = index
        self.constant = constant
    
    def compute(self, x):
        x = x.copy()
        x[self.index] = self.constant
        return x
    
    def gradient(self, output_grad, x):
        x_grad = output_grad.copy()
        x_grad[self.index] = 0
        return x_grad

# define the class for the negative operation
class Neg(Operation):
    def compute(self, x):
        return -x
    
    def gradient(self, output_grad, x):
        return -output_grad
    
# define the class for the addition operation
class Add(BCOperation):
    def compute(self, x, y):
        return x + y
    
    def gradient(self, output_grad, x, y):
        return output_grad, output_grad
    
# define the class for the constant addition operation
class AddConstant(Operation):
    def __init__(self, constant):
        self.constant = constant
    
    def compute(self, x):
        return x + self.constant
    
    def gradient(self, output_grad, x):
        return output_grad
    
# define the class for the subtraction operation
class Sub(BCOperation):
    def compute(self, x, y):
        return x - y
    
    def gradient(self, output_grad, x, y):
        return output_grad, -output_grad

# define the class for the constant subtraction operation
class SubConstant(Operation):
    def __init__(self, constant):
        self.constant = constant
    
    def compute(self, x):
        return x - self.constant
    
    def gradient(self, output_grad, x):
        return output_grad

# define the class for the right constant subtraction operation
class RSubConstant(Operation):
    def __init__(self, constant):
        self.constant = constant
    
    def compute(self, x):
        return self.constant - x
    
    def gradient(self, output_grad, x):
        return -output_grad

# define the class for the multiplication operation
class Mul(BCOperation):
    def compute(self, x, y):
        return x * y
    
    def gradient(self, output_grad, x, y):
        return y * output_grad, x * output_grad
    
# define the class for the constant multiplication operation
class MulConstant(Operation):
    def __init__(self, constant):
        self.constant = constant
    
    def compute(self, x):
        return x * self.constant
    
    def gradient(self, output_grad, x):
        return self.constant * output_grad
    
# define the class for the division operation
class Div(BCOperation):
    def compute(self, x, y):
        return x / y
    
    def gradient(self, output_grad, x, y):
        #print(output_grad, x, y)
        return output_grad / y, -x * output_grad / (y * y)
    
# define the class for the constant division operation
class DivConstant(Operation):
    def __init__(self, constant):
        self.constant = constant
    
    def compute(self, x):
        return x / self.constant
    
    def gradient(self, output_grad, x):
        return output_grad / self.constant
    
# define the class for the right constant division operation
class RDivConstant(Operation):
    def __init__(self, constant):
        self.constant = constant
    
    def compute(self, x):
        return self.constant / x
    
    def gradient(self, output_grad, x):
        return -self.constant * output_grad / (x * x)
    
# define the class for the matrix multiplication operation
class MatMul(Operation):
    def compute(self, x, y):
        return x @ y
    
    def gradient(self, output_grad, x, y):
        return output_grad @ y.T, x.T @ output_grad
    
# define the class for the sum operation
class Sum(Operation):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def compute(self, x):
        return x.sum(axis=self.axis, keepdims=self.keepdims)
    
    def gradient(self, output_grad, x):
        return np.full_like(x, output_grad)
    
# define the class for the sum of tensors in a list
class LSum(Operation):
    def compute(self, *x):
        return np.sum([data for data in x])
    
    def gradient(self, output_grad, *x):
        return [output_grad for data in x]

# define the class for the mean operation
class Mean(Operation):
    def __init__(self, axis=None, keepdims=False):
        self.axis = axis
        self.keepdims = keepdims

    def compute(self, x):
        return x.mean(axis=self.axis, keepdims=self.keepdims)
    
    def gradient(self, output_grad, x):
        return np.full_like(x, output_grad / x.size)
    
# define the class for the transpose operation
class Transpose(Operation):
    def compute(self, x):
        return x.T
    
    def gradient(self, output_grad, x):
        return output_grad.T

# define the class for the reshape operation
class Reshape(Operation):
    def __init__(self, shape):
        self.shape = shape
    
    def compute(self, x):
        return x.reshape(self.shape)
    
    def gradient(self, output_grad, x):
        return output_grad.reshape(x.shape)
    
# define the class for the concatenation of a list of tensors operation
class Concat(Operation):
    def __init__(self, axis):
        self.axis = axis
    
    # x is a list of arrays
    def compute(self, *x):
        return np.concatenate(x, axis=self.axis)
    
    def gradient(self, output_grad, *x):
        axis = len(output_grad.shape) - 1 if self.axis == -1 else self.axis
        idx = 0
        grads = []
        for i in range(len(x)):
            grads.append(output_grad[
                (slice(None),)*axis + (slice(idx, idx+x[i].shape[axis]),)])
            idx += x[i].shape[axis]
        return tuple(grads)
        
    
# define the class for the power operation
class Power(Operation):
    def compute(self, x, y):
        return x ** y
    
    def gradient(self, output_grad, x, y):
        x_grad = y * (x ** (y - 1)) * output_grad
        y_grad = (x ** y) * np.log(x) * output_grad
        return x_grad, y_grad
    
# define the class for the constant power operation
class PowerConstant(Operation):
    def __init__(self, constant):
        self.constant = constant
    
    def compute(self, x):
        return x ** self.constant
    
    def gradient(self, output_grad, x):
        return self.constant * (x ** (self.constant - 1)) * output_grad

# define the class for the right constant power operation
class RPowerConstant(Operation):
    def __init__(self, constant):
        self.constant = constant
    
    def compute(self, x):
        return self.constant ** x
    
    def gradient(self, output_grad, x):
        return (self.constant ** x) * np.log(self.constant) * output_grad
    
# define the class for the exponential operation
class Exp(Operation):
    def compute(self, x):
        return np.exp(x)
    
    def gradient(self, output_grad, x):
        return np.exp(x) * output_grad
    
# define the class for the log operation
class Log(Operation):
    def compute(self, x):
        return np.log(x)
    
    def gradient(self, output_grad, x):
        return output_grad / x
    
# define the class for sigmoid operation
class Sigmoid(Operation):
    def compute(self, x):
        #return 1/(1+np.exp(-x))
        out = np.empty_like(x)
        mask = x > 0
        out[mask] = 1/(1+np.exp(-x[mask]))
        f =  np.exp(x[~mask])
        out[~mask] = f / (f+1)
        return out
        
    
    def gradient(self, output_grad, x):
        f = self.compute(x)
        return f*(1-f)*output_grad
