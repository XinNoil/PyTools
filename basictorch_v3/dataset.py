from abc import ABC, abstractmethod
import torch, os
from mtools import list_con, join_path
from torch.utils.data import DataLoader
from torchvision import datasets, transforms

class BDatasets(ABC):
    """
    An abstract interface for compiled sequence.
    """
    def __init__(self, is_shuffle, is_validate, split_val, batch_size, train_dataset=None, test_dataset=None, val_dataset=None, **kwargs):
        super().__init__()
        self.is_shuffle = is_shuffle
        self.is_validate = is_validate
        self.split_val = split_val
        self.batch_size = batch_size
        self.set_train_dataset(train_dataset, val_dataset, **kwargs)
        self.set_test_dataset(test_dataset, **kwargs)
        self.print_shape()
        self.ds_dict = {'train':self.train_dataset, 'test':self.test_dataset, 'val':self.val_dataset}

    def get_data_loader(self, ds_name, batch_size=None):
        if ds_name == 'train':
            batch_size = batch_size if batch_size else self.batch_size
            shuffle = self.is_shuffle
        else:
            batch_size = batch_size if batch_size else 1000
            shuffle = False
        ds = self.ds_dict[ds_name]
        if hasattr(self.ds_dict[ds_name], 'get_data_loader'):
            return self.ds_dict[ds_name].get_data_loader(batch_size, shuffle)
        else:
            return DataLoader(ds, batch_size, shuffle=shuffle)

    def set_train_dataset(self, train_dataset=None, val_dataset=None, **kwargs):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    
    def set_test_dataset(self, test_dataset=None, **kwargs):
        self.test_dataset = test_dataset
    
    def print_shape(self, ds_names=['train_dataset', 'test_dataset', 'val_dataset']):
        for ds_name in ds_names:
            if ds_name in self.__dict__:
                ds = self.__dict__[ds_name]
                if hasattr(ds, 'tensors'):
                    for tensor, tensor_name in zip(ds.tensors, self.tensor_names):
                        print('%s.%s: '%(ds_name,tensor_name), tensor.shape)
        print('')

class MDataLoader(object):
    def __init__(self, *dataLoaders):
        self.dataLoaders = dataLoaders
    
    def __len__(self):
        return min([len(d) for d in self.dataLoaders])
        
    def __iter__(self):
      for batch in zip(*self.dataLoaders):
          yield list_con(batch)

def merge_batch(batch):
    return tuple(torch.cat(tuple(_[i] for _ in batch), dim=0) for i in range(len(batch[0])))

class STDataLoader(object):
    def __init__(self, s_dataLoaders, t_dataLoaders):
        self.s_dataLoaders = s_dataLoaders
        self.t_dataLoaders = t_dataLoaders
    
    def __len__(self):
        return min([len(d) for d in [*self.s_dataLoaders, *self.t_dataLoaders]])
        
    def __iter__(self):
        for batch in zip(*self.s_dataLoaders, *self.t_dataLoaders):
            yield batch[:len(self.s_dataLoaders)], batch[len(self.s_dataLoaders):]

dataset_path = os.environ['DATA_PATH']

class ImageDataset(BDatasets):
    def __init__(self, is_shuffle, is_validate, split_val, batch_size, train_dataset=None, test_dataset=None, val_dataset=None, **kwargs):
        super().__init__(is_shuffle, is_validate, split_val, batch_size, train_dataset, test_dataset, val_dataset, **kwargs)

    def set_train_dataset(self, train_dataset=None, val_dataset=None, **kwargs):
        self.train_dataset = train_dataset
        self.val_dataset = val_dataset
    
    def set_test_dataset(self, test_dataset=None, **kwargs):
        self.test_dataset = test_dataset

def dataloader(dataset, input_size, batch_size, split='train'):
    transform = transforms.Compose([transforms.Resize((input_size, input_size)), transforms.ToTensor(), transforms.Normalize(mean=(0.5, 0.5, 0.5), std=(0.5, 0.5, 0.5))])
    if dataset == 'mnist':
        transform = transforms.Compose([transforms.ToTensor(), transforms.Normalize(mean=(0.5,), std=(0.5,))])
        data_loader = DataLoader(
            datasets.MNIST(join_path(dataset_path, 'mnist'), train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'fashion-mnist':
        data_loader = DataLoader(
            datasets.FashionMNIST(join_path(dataset_path, 'fashion-mnist'), train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'cifar10':
        data_loader = DataLoader(
            datasets.CIFAR10(join_path(dataset_path, 'cifar10'), train=True, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'svhn':
        data_loader = DataLoader(
            datasets.SVHN(join_path(dataset_path, 'svhn'), split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'stl10':
        data_loader = DataLoader(
            datasets.STL10(join_path(dataset_path, 'stl10'), split=split, download=True, transform=transform),
            batch_size=batch_size, shuffle=True)
    elif dataset == 'lsun-bed':
        data_loader = DataLoader(
            datasets.LSUN(join_path(dataset_path, 'lsun'), classes=['bedroom_train'], transform=transform),
            batch_size=batch_size, shuffle=True)

    return data_loader