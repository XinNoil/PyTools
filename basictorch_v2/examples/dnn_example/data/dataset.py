import numpy as np
import random
from basictorch_v2.dataset import BDatasets
from datahub.db import DB
from torch.utils.data import Dataset, DataLoader
from mtools import list_ind

class WiFiDataset(Dataset):
    def __init__(self, db=None, dataset=None, **kwargs) -> None:
        super().__init__()
        if db is not None:
            self.features  = db.get_feature()
            self.labels  = db.get_label()
            self.index_map =  list(range(len(db)))
            if kwargs.get('shuffle', True):
                random.shuffle(self.index_map)
            if kwargs.get('data_usage', 1)<1:
                use_num = int(len(self.index_map)*kwargs.get('data_usage', 1))
                p = np.random.permutation(len(self.index_map))
                self.index_map = list_ind(self.index_map, p[:use_num])
        else:
            self.__dict__ = dataset.__dict__.copy()
            val_num = int(len(self.index_map)*kwargs.get('data_usage', 0.1))
            p = np.random.permutation(len(self.index_map))
            self.index_map = list_ind(self.index_map, p[:val_num])
            dataset.index_map = list_ind(dataset.index_map, p[val_num:])
    
    def __getitem__(self, item):
        return self.features[self.index_map[item]].astype(np.float32), self.labels[self.index_map[item]].astype(np.float32)

    def __len__(self):
        return len(self.index_map)
    
    def get_data_loader(self, batch_size=1000, shuffle=False):
        return DataLoader(self, batch_size, shuffle=shuffle)
    
    def get_feature_dim(self):
        return self.features.shape[1]
    
    def get_label_dim(self):
        return self.labels.shape[1]
    
    def print_shape(self):
        print('length: ', len(self))

class Datasets(BDatasets):
    def __init__(self, is_shuffle, is_validate, split_val, batch_size, train_dataset, test_dataset, **kwargs):
        super().__init__(is_shuffle, is_validate, split_val, batch_size, train_dataset, test_dataset, **kwargs)
    
    def set_train_dataset(self, train_dataset=None, val_dataset=None, **kwargs):
        if val_dataset is None:
            val_dataset = WiFiDataset(dataset=train_dataset, split_val=self.split_val)
        return super().set_train_dataset(train_dataset, val_dataset, **kwargs)