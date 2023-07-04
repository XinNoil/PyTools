import torch
import mtools.monkey as mk
from torch.utils.data import DataLoader, Dataset

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

def get_test_dataset(test_dataset, batch_size):
    if issubclass(type(test_dataset), DataLoader):
        return test_dataset.dataset, test_dataset
    elif issubclass(type(test_dataset), Dataset):
        return test_dataset, DataLoader(test_dataset, batch_size=batch_size, shuffle=False)
    else:
        raise TypeError(f"Unexpected test_dataset type :{type(test_dataset)}")

def get_dataset_property(dataset, name):
    if hasattr(dataset, name):
        return getattr(dataset, name)
    else:
        return get_dataset_property(dataset.dataset, name)

class Obj(object):
    def __init__(self, obj_dict) -> None:
        self.__dict__ = obj_dict
    
    def get(self, __name, __default=None):
        return getattr(self, __name, __default)
    
    def keys(self):
        return self.__dict__.keys()

def dictConfig_to_dict(cfg):
    return Obj(dict(zip(cfg.keys(), cfg.values())))