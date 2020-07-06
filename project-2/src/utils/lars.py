import torch
from torch.optim.optimizer import Optimizer

class LARS(Optimizer):
    """
    Layer-wise rate scaling (LARS) for SGD

    args:
    params: parameters to optimize
    lr: base learning rate
    momentum: momentum factor
    weight_decay: L2 penalty
    eta: LARS coefficient
    max_epoch: maximum training epoch to determine polynomial LR decay

    usage:
    optimizer = LARS(model.parameters(), lr=0.1, eta=1e-3)
    optimizer.zero_grad()
    loss_fn(model(input), target).backward()
    optimizer.step()
    """

    def __init__(self, params, lr=0.1, momentum=0.9, weight_decay=0.0005, eta=0.001, max_epoch=200):
        self.epoch = 0
        defaults = dict(lr=lr, momentum=momentum, weight_decay=weight_decay, eta=eta, max_epoch=max_epoch)
        super(LARS, self).__init__(params, defaults)

    def step(self, epoch=None, closure=None):
        """
        Perform single optimization step
        
        args:
        closure: reevaluates the model and return the loss
        epoch: current epoch to calculate polynomial LR decay schedule
        """
        loss = None
        if closure is not None:
            loss = closure()
        if epoch is None:
            epoch = self.epoch
            self.epoch += 1

        for group in self.param_groups:
            weight_decay = group['weight_decay']
            momentum = group['momentum']
            eta = group['eta']
            lr = group['lr']
            max_epoch = group['max_epoch']

            for param in group['params']:
                if param.grad is None:
                    continue

                param_state = self.state[param]
                dparam = param.grad.data

                weight_norm = torch.norm(param.data)
                grad_norm = torch.norm(dparam)

                # Global LR computed on polynomial decay schedule
                decay = (1 - float(epoch) / max_epoch) ** 2
                global_lr = lr * decay

                # Compute local learning rate for this layer
                local_lr = eta * weight_norm / (grad_norm + weight_decay * weight_norm)

                # Update the momentum term
                actual_lr = local_lr * global_lr

                if 'momentum_buffer' not in param_state:
                    buffer = param_state['momentum_buffer'] = \
                            torch.zeros_like(param.data)
                else:
                    buffer = param_state['momentum_buffer']
                buffer.mul_(momentum).add_(dparam + weight_decay * param.data, alpha=actual_lr)
                param.data.add_(-buffer)

        return loss


        