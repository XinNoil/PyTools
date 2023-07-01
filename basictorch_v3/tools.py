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

def freeze_model(model):
    for p in model.parameters():
        p.requires_grad = False

def unfreeze_model(model):
    for p in model.parameters():
        p.requires_grad = True

def data_to_device(batch_data, device=None):
    device = mk.get_current_device() if device is None else device
    if isinstance(batch_data, torch.Tensor):
        return batch_data.to(device)
    else:
        return tuple(data_to_device(item, device) for item in batch_data)

def count_parameters(net):
    return sum(p.numel() for p in net.parameters() if p.requires_grad)