from abc import ABC, abstractmethod
import numpy as np
from scipy.optimize import fmin_l_bfgs_b as minimize
from agtensors import *


class Parameter(AGTensor):
    def __init__(self, data, dtype=np.float32, regularize=True):
        super().__init__(data.astype(dtype))
        self.regularize = regularize

# define an abstract class for the layers
class Module(ABC):
    def __init__(self):
        self.params_list = []

    @abstractmethod
    def forward(self, *inputs):
        raise NotImplementedError
    
    def __call__(self, *inputs):
        return self.forward(*inputs)
    
    def __setattr__(self, name, value):
        if isinstance(value, Parameter):
            self.params_list.append(value)
        if isinstance(value, Module):
            self.params_list += value.params_list
        super().__setattr__(name, value)
    
    def register_layers(self, layers):
        for layer in layers:
            self.params_list += layer.params_list

    # return a numpy array with the parameters of the module
    def get_params(self):
        return np.concatenate([p.data.flatten() for p in self.params_list])
    
    # set the parameters of the module
    def set_params(self, params):
        idx = 0
        for p in self.params_list:
            p.data[:] = params[idx:idx+p.data.size].reshape(p.data.shape)
            idx += p.data.size

    # return a numpy array with the gradients of the module
    def get_grads(self):
        return np.concatenate([p.grad.flatten() for p in self.params_list])
    
    # for Marko :)
    def weights(self):
        # contatenates weights and biases for each layer
        weights = []
        w = self.params_list[::2]
        b = self.params_list[1::2]
        for i in range(len(w)):
            wb = np.concatenate((w[i].data, b[i].data), axis=0)
            weights.append(wb)
        return weights
    
    def predict(self, X):
        if not isinstance(X, AGTensor):
            X = AGTensor(X.astype(np.float32))
        return self.forward(X).data.squeeze()

def l2(params, lambda_):
    reg = 0
    for p in params:
        fac = np.sqrt(6 / (sum(p.data.shape)))
        reg = reg + 0.5*((p**2).sum())*fac
    return lambda_ * reg

class Learner:
    def __init__(self, model, model_args, loss, lambda_):
        self.model = model
        self.model_args = model_args
        self.loss = loss
        self.lambda_ = lambda_
    
    def optim_function(self, model, params, X, y):
        model.set_params(params)
        y_pred = model(X)
        loss = self.loss(y_pred, y)
        if self.lambda_ > 0:
            loss = loss + l2(self.params2reg, self.lambda_)
        loss.zero_grad()
        loss.backward()
        grads = model.get_grads()
        return loss.data, grads

    def fit(self, X, y, max_iter=500):
        # initialize model
        model = self.model(**self.model_args)
        if self.lambda_ > 0:
            self.params2reg = [
                param for param in model.params_list if param.regularize]


        if not isinstance(X, AGTensor):
            X = AGTensor(X.astype(np.float32), constant=True)
        if not isinstance(y, AGTensor):
            y = AGTensor(y.astype(np.float32), constant=True)
        params = model.get_params()

        res = minimize(
            lambda x: self.optim_function(model, x, X, y),
            params,
            fprime=None, approx_grad=False,
            maxiter=max_iter
        )
        params = res[0]
        print(res[2]["task"])
        model.set_params(params)
        return model
