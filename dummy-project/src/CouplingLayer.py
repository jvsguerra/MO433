import torch
import torch.nn as nn 

from ResNet import ResNet
from misc import checkerboard_mask

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
        if self.mask == 1: # channel-wise mask
            cin //= 2
        
        self.st_net = ResNet(cin, cmid, 2 * cin, nblocks=nblocks, 
                            kernel=3, padding=1, 
                            double_after_norm=(self.mask == 0) # checkboard mask
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
                sldj += s.view(s.size(0), -1).sum(-1)
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
                sldj += s.view(s.size(0), -1).sum(-1)

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