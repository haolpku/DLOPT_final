class AdagradOptimizer:
    def __init__(self, parameters, lr=0.01, epsilon=1e-10):
        self.parameters = list(parameters)
        self.lr = lr
        self.epsilon = epsilon
        self.squared_gradients = [torch.zeros_like(p) for p in self.parameters]

    def step(self):
        with torch.no_grad():
            for param, squared_gradient in zip(self.parameters, self.squared_gradients):
                if param.grad is not None:
                    squared_gradient.add_(param.grad.pow(2))
                    std = squared_gradient.sqrt().add_(self.epsilon)
                    param.addcdiv_(-self.lr, param.grad, std)


class RMSPropOptimizer:
    def __init__(self, parameters, lr=0.01, alpha=0.99, epsilon=1e-8):
        self.parameters = list(parameters)
        self.lr = lr
        self.alpha = alpha
        self.epsilon = epsilon
        self.squared_gradients = [torch.zeros_like(p) for p in self.parameters]

    def step(self):
        with torch.no_grad():
            for param, squared_gradient in zip(self.parameters, self.squared_gradients):
                if param.grad is not None:
                    squared_gradient.mul_(self.alpha).addcmul_(1 - self.alpha, param.grad, param.grad)
                    std = squared_gradient.sqrt().add(self.epsilon)
                    param.addcdiv_(-self.lr, param.grad, std)


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

    def step(self):
        self.t += 1
        lr_t = self.lr * (1 - self.beta2 ** self.t) ** 0.5 / (1 - self.beta1 ** self.t)
        with torch.no_grad():
            for param, moment, velocity in zip(self.parameters, self.moments, self.velocities):
                if param.grad is not None:
                    moment.mul_(self.beta1).add_(1 - self.beta1, param.grad)
                    velocity.mul_(self.beta2).addcmul_(1 - self.beta2, param.grad, param.grad)
                    param.addcdiv_(-lr_t, moment, velocity.sqrt().add(self.epsilon))


class MomentumSGDOptimizer:
    def __init__(self, parameters, lr=0.01, momentum=0.9):
        self.parameters = list(parameters)
        self.lr = lr
        self.momentum = momentum
        self.velocities = [torch.zeros_like(p) for p in self.parameters]

    def step(self):
        with torch.no_grad():
            for param, velocity in zip(self.parameters, self.velocities):
                if param.grad is not None:
                    velocity.mul_(self.momentum).add_(param.grad)
                    param.add_(-self.lr, velocity)


class SGDOptimizer:
    def __init__(self, parameters, lr=0.01):
        self.parameters = list(parameters)
        self.lr = lr

    def step(self):
        with torch.no_grad():
            for param in self.parameters:
                if param.grad is not None:
                    param.add_(-self.lr, param.grad)
