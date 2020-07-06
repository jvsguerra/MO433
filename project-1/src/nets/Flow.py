import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np

from nets.CouplingLayer import CouplingLayer

mask_checkboard = 0
mask_channelwise = 1


def squeeze_2x2(x, reverse=False, alt_order=False):
    """For each spatial position, a sub-volume of shape `1x1x(N^2 * C)`,
    reshape into a sub-volume of shape `NxNxC`, where `N = block_size`.
    Adapted from:
        https://github.com/tensorflow/models/blob/master/research/real_nvp/real_nvp_utils.py
    See Also:
        - TensorFlow nn.depth_to_space: https://www.tensorflow.org/api_docs/python/tf/nn/depth_to_space
        - Figure 3 of RealNVP paper: https://arxiv.org/abs/1605.08803
    Args:
        x (torch.Tensor): Input tensor of shape (B, C, H, W).
        reverse (bool): Whether to do a reverse squeeze (unsqueeze).
        alt_order (bool): Whether to use alternate ordering.
    """
    block_size = 2
    if alt_order:
        n, c, h, w = x.size()

        if reverse:
            if c % 4 != 0:
                raise ValueError('Number of channels must be divisible by 4, got {}.'.format(c))
            c //= 4
        else:
            if h % 2 != 0:
                raise ValueError('Height must be divisible by 2, got {}.'.format(h))
            if w % 2 != 0:
                raise ValueError('Width must be divisible by 4, got {}.'.format(w))
        # Defines permutation of input channels (shape is (4, 1, 2, 2)).
        squeeze_matrix = torch.tensor([[[[1., 0.], [0., 0.]]],
                                       [[[0., 0.], [0., 1.]]],
                                       [[[0., 1.], [0., 0.]]],
                                       [[[0., 0.], [1., 0.]]]],
                                      dtype=x.dtype,
                                      device=x.device)
        perm_weight = torch.zeros((4 * c, c, 2, 2), dtype=x.dtype, device=x.device)
        for c_idx in range(c):
            slice_0 = slice(c_idx * 4, (c_idx + 1) * 4)
            slice_1 = slice(c_idx, c_idx + 1)
            perm_weight[slice_0, slice_1, :, :] = squeeze_matrix
        shuffle_channels = torch.tensor([c_idx * 4 for c_idx in range(c)]
                                        + [c_idx * 4 + 1 for c_idx in range(c)]
                                        + [c_idx * 4 + 2 for c_idx in range(c)]
                                        + [c_idx * 4 + 3 for c_idx in range(c)])
        perm_weight = perm_weight[shuffle_channels, :, :, :]

        if reverse:
            x = F.conv_transpose2d(x, perm_weight, stride=2)
        else:
            x = F.conv2d(x, perm_weight, stride=2)
    else:
        b, c, h, w = x.size()
        x = x.permute(0, 2, 3, 1)

        if reverse:
            if c % 4 != 0:
                raise ValueError('Number of channels {} is not divisible by 4'.format(c))
            x = x.view(b, h, w, c // 4, 2, 2)
            x = x.permute(0, 1, 4, 2, 5, 3)
            x = x.contiguous().view(b, 2 * h, 2 * w, c // 4)
        else:
            if h % 2 != 0 or w % 2 != 0:
                raise ValueError('Expected even spatial dims HxW, got {}x{}'.format(h, w))
            x = x.view(b, h // 2, 2, w // 2, 2, c)
            x = x.permute(0, 1, 3, 5, 2, 4)
            x = x.contiguous().view(b, h // 2, w // 2, c * 4)

        x = x.permute(0, 3, 1, 2)

    return x


class Flow(nn.Module):
    """
    args:
    nscales: Number of scales in Flow
    cin: Input channels
    cmid: Intermediate layers
    nblocks: Residual blocks in the s and t network of Coupling Layers
    """
    def __init__(self, nscales=2, cin=3, cmid=64, nblocks=8, device='cpu'):
        super(Flow, self).__init__()

        # Bounds
        self.device = device
        self.bounds = torch.tensor([0.9], dtype=torch.float32).to(self.device)

        # Define Flow
        self.flows = _Flow(0, nscales, cin, cmid, nblocks)

    def forward(self, x, reverse=False):
        sum_log_det = None
        if not reverse:
            # Expect inputs in [0, 1]
            # FIXME: ver se precisa
            if x.min() < 0 or x.max() > 1:
                raise ValueError(f'Expected x in [0, 1], got x with min/max {x.min()}/{x.max()}')
        
            # De-quantize
            x = self.dequantize(x)
            
            # Convert to logit
            x = self.logit(x)

            # Get log-determinant of Jacobian matrix
            log_det = F.softplus(x) + F.softplus(-x) - F.softplus((1. - self.bounds).log() - self.bounds.log())
            sum_log_det = log_det.reshape(log_det.size(0), -1).sum(-1)
        
        # Flow
        x, sum_log_det = self.flows(x, sum_log_det, reverse)

        return x, sum_log_det

    def logit(self, x):
        y = (2 * x - 1) * self.bounds
        y = (y + 1) / 2
        y = y.log() - (1.0 - y).log()
        return y

    def dequantize(self, x, method='uniform'):
        # y = (x * 255.0 + torch.rand_like(x)) / 256.0
        if method == 'uniform':
            # y = (x * 255.0 + torch.distributions.Uniform(0.0, 1.0).sample(x.shape).to(misc.device)) / 256.0
            y = (x * 255.0 + torch.distributions.Uniform(0.0, 1.0).sample(x.shape).to(self.device)) / 256.0
        return y

# TODO: Fixme

class _Flow(nn.Module):
    """Recursive builder for a `RealNVP` model.
    Each `_FlowBuilder` corresponds to a single scale in `RealNVP`,
    and the constructor is recursively called to build a full `RealNVP` model.
    Args:
        scale_idx (int): Index of current scale.
        num_scales (int): Number of scales in the RealNVP model.
        in_channels (int): Number of channels in the input.
        mid_channels (int): Number of channels in the intermediate layers.
        num_blocks (int): Number of residual blocks in the s and t network of
            `Coupling` layers.
    """
    def __init__(self, scale_idx, num_scales, in_channels, mid_channels, num_blocks):
        super(_Flow, self).__init__()

        self.is_last_block = scale_idx == num_scales - 1

        self.in_couplings = nn.ModuleList([
            CouplingLayer(in_channels, mid_channels, num_blocks, mask_checkboard, reverse_mask=False),
            CouplingLayer(in_channels, mid_channels, num_blocks, mask_checkboard, reverse_mask=True),
            CouplingLayer(in_channels, mid_channels, num_blocks, mask_checkboard, reverse_mask=False)
        ])

        if self.is_last_block:
            self.in_couplings.append(
                CouplingLayer(in_channels, mid_channels, num_blocks, mask_checkboard, reverse_mask=True))
        else:
            self.out_couplings = nn.ModuleList([
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, mask_channelwise, reverse_mask=False),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, mask_channelwise, reverse_mask=True),
                CouplingLayer(4 * in_channels, 2 * mid_channels, num_blocks, mask_channelwise, reverse_mask=False)
            ])
            self.next_block = _Flow(scale_idx + 1, num_scales, 2 * in_channels, 2 * mid_channels, num_blocks)

    def forward(self, x, sldj, reverse=False):
        if reverse:
            if not self.is_last_block:
                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)

                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, reverse=False)
                for coupling in reversed(self.out_couplings):
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

            for coupling in reversed(self.in_couplings):
                x, sldj = coupling(x, sldj, reverse)
        else:
            for coupling in self.in_couplings:
                x, sldj = coupling(x, sldj, reverse)

            if not self.is_last_block:
                # Squeeze -> 3x coupling (channel-wise)
                x = squeeze_2x2(x, reverse=False)
                for coupling in self.out_couplings:
                    x, sldj = coupling(x, sldj, reverse)
                x = squeeze_2x2(x, reverse=True)

                # Re-squeeze -> split -> next block
                x = squeeze_2x2(x, reverse=False, alt_order=True)
                x, x_split = x.chunk(2, dim=1)
                x, sldj = self.next_block(x, sldj, reverse)
                x = torch.cat((x, x_split), dim=1)
                x = squeeze_2x2(x, reverse=True, alt_order=True)

        return x, sldj


class Loss(nn.Module):
    """ NLL Loss """
    def  __init__(self, k=32*32):
        super(Loss, self).__init__()
        self.k = k
    
    def forward(self, z, sum_log_det):
        prior = -0.5 * (z ** 2 + np.log(2 * np.pi))
        prior = prior.reshape(z.size(0), -1).sum(-1) - np.log(self.k) * np.prod(z.size()[1:])
        
        log_likelihood = prior + sum_log_det
        negative_log_likelihood = -log_likelihood.mean()
        return negative_log_likelihood


class LossMeter(object):

    def __init__(self):
        # self.val = 0.0
        self.avg = 0.0
        self.sum = 0.0
        self.count = 0.0
    
    def update(self, val, n):
        # self.val = val
        self.sum += val * n
        self.count += n
        self.avg = self.sum / self.count