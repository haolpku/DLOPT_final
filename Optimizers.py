import torch 
class SGDOptimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        with torch.no_grad():
            for param in self.parameters:
                if param.grad is not None:
                    param -= self.lr * param.grad

class MomentumSGDOptimizer:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.parameters]

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        with torch.no_grad():
            for param, velocity in zip(self.parameters, self.velocities):
                if param.grad is not None:
                    velocity *= self.momentum
                    velocity += self.lr * param.grad
                    param -= velocity

class AdagradOptimizer:
    def __init__(self, parameters, lr=0.01, epsilon=1e-10):
        self.parameters = list(parameters)
        self.lr = lr
        self.epsilon = epsilon
        self.squared_gradients = [torch.zeros_like(p) for p in self.parameters]

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        with torch.no_grad():
            for param, squared_gradient in zip(self.parameters, self.squared_gradients):
                if param.grad is not None:
                    squared_gradient += param.grad ** 2
                    param -= self.lr * param.grad / (squared_gradient.sqrt() + self.epsilon)

class RMSPropOptimizer:
    def __init__(self, parameters, lr=0.01, alpha=0.99, epsilon=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.alpha = alpha
        self.epsilon = epsilon
        self.squared_gradients = [torch.zeros_like(p) for p in self.parameters]

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        with torch.no_grad():
            for param, squared_gradient in zip(self.parameters, self.squared_gradients):
                if param.grad is not None:
                    squared_gradient *= self.alpha
                    squared_gradient += (1 - self.alpha) * param.grad ** 2
                    param -= self.lr * param.grad / (squared_gradient.sqrt() + self.epsilon)

class AdamOptimizer:
    def __init__(self, parameters, lr=0.001, beta1=0.9, beta2=0.999, epsilon=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.beta1 = beta1
        self.beta2 = beta2
        self.epsilon = epsilon
        self.moments = [torch.zeros_like(p) for p in self.parameters]
        self.velocities = [torch.zeros_like(p) for p in self.parameters]
        self.t = 0

    def zero_grad(self):
        for param in self.parameters:
            if param.grad is not None:
                param.grad.zero_()

    def step(self):
        self.t += 1
        lr_t = self.lr * (1 - self.beta2 ** self.t) ** 0.5 / (1 - self.beta1 ** self.t)
        with torch.no_grad():
            for param, moment, velocity in zip(self.parameters, self.moments, self.velocities):
                if param.grad is not None:
                    moment *= self.beta1
                    moment += (1 - self.beta1) * param.grad
                    velocity *= self.beta2
                    velocity += (1 - self.beta2) * param.grad ** 2
                    param -= lr_t * moment / (velocity.sqrt() + self.epsilon)
