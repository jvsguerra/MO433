import torch
import torch.nn as nn 

from nets.ResNet import ResNet

mask_checkboard = 0
mask_channelwise = 1

def checkerboard_mask(height, width, reverse=False, dtype=torch.float32, device=None, requires_grad=False):
    """Get a checkerboard mask, such that no two entries adjacent entries
    have the same value. In non-reversed mask, top-left entry is 0.
    Args:
        height (int): Number of rows in the mask.
        width (int): Number of columns in the mask.
        reverse (bool): If True, reverse the mask (i.e., make top-left entry 1).
            Useful for alternating masks in RealNVP.
        dtype (torch.dtype): Data type of the tensor.
        device (torch.device): Device on which to construct the tensor.
        requires_grad (bool): Whether the tensor requires gradient.
    Returns:
        mask (torch.tensor): Checkerboard mask of shape (1, 1, height, width).
    """
    checkerboard = [[((i % 2) + j) % 2 for j in range(width)] for i in range(height)]
    mask = torch.tensor(checkerboard, dtype=dtype, device=device, requires_grad=requires_grad)

    if reverse:
        mask = 1 - mask

    # Reshape to (1, 1, height, width) for broadcasting with tensors of shape (B, C, H, W)
    mask = mask.reshape(1, 1, height, width)

    return mask

class CouplingLayer(nn.Module):
    """
    cin: channels in
    cmid: channels in s and t network
    nblocks: number of residual blocks in s and t network
    mask: checkboard (0) or channelwise (1) # FIXME
    reverse_mask: whether to reverse the mask, useful for alternating masks. # FIXME
    """
    def __init__(self, cin, cmid, nblocks, mask, reverse_mask):
        super(CouplingLayer, self).__init__()

        # Save info
        self.mask = mask
        self.reverse_mask = reverse_mask

        # Build scale and translate network
        if self.mask == mask_channelwise:
            cin //= 2
        
        self.st_net = ResNet(cin, cmid, 2 * cin, nblocks=nblocks, 
                            kernel=3, padding=1, 
                            double_after_norm=(self.mask == mask_checkboard)
                            )
        
        # Learn scale for s
        self.rescale = nn.utils.weight_norm(Rescale(cin)) #TODO: Create Rescale
    
    def forward(self, x, sldj=None, reverse=True):
        if self.mask == 0:
            # Checkerboard mask
            b = checkerboard_mask(x.size(2), x.size(3), self.reverse_mask, device=x.device)
            x_b = x * b
            st = self.st_net(x_b)
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))
            s = s * (1 - b)
            t = t * (1 - b)

            # Scale and translate
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = x * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x = (x + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.reshape(s.size(0), -1).sum(-1)
        else:
            # Channel-wise mask
            if self.reverse_mask:
                x_id, x_change = x.chunk(2, dim=1)
            else:
                x_change, x_id = x.chunk(2, dim=1)

            st = self.st_net(x_id)
            s, t = st.chunk(2, dim=1)
            s = self.rescale(torch.tanh(s))

            # Scale and translate
            if reverse:
                inv_exp_s = s.mul(-1).exp()
                if torch.isnan(inv_exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = x_change * inv_exp_s - t
            else:
                exp_s = s.exp()
                if torch.isnan(exp_s).any():
                    raise RuntimeError('Scale factor has NaN entries')
                x_change = (x_change + t) * exp_s

                # Add log-determinant of the Jacobian
                sldj += s.reshape(s.size(0), -1).sum(-1)

            if self.reverse_mask:
                x = torch.cat((x_id, x_change), dim=1)
            else:
                x = torch.cat((x_change, x_id), dim=1)

        return x, sldj


class Rescale(nn.Module):
    """Per-channel rescaling. Need a proper `nn.Module` so we can wrap it
    with `torch.nn.utils.weight_norm`.
    Args:
        cin (int): Number of channels in the input.
    """
    def __init__(self, cin):
        super(Rescale, self).__init__()
        self.weight = nn.Parameter(torch.ones(cin, 1, 1))

    def forward(self, x):
        x = self.weight * x
        return x