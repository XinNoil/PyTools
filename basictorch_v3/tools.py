import torch
import mtools.monkey as mk

def item_losses(losses):
    for loss in losses:
        losses[loss] = losses[loss].item()
    return losses

def n2t(num, tensortype=torch.FloatTensor, device=None, **kwargs):
    return tensortype(num, **kwargs).to(mk.get_current_device() if device is None else device)

def t2n(tensor):
    return tensor.detach().cpu().numpy()