# -*- coding: UTF-8 -*-
import os, sys
from mtools import monkey as mk
from .ITrainer import ITrainer
from .IDataset import IDataset
from .IModel import IModel
from torch.utils.data import random_split, DataLoader
import numpy as np
np.set_printoptions(precision=5, suppress=True, formatter={'float_kind': '{:f}'.format})

import torch
import logging
import ipdb as pdb

from tqdm import tqdm

log = logging.getLogger('BaseTrainer')

class BaseTrainer(ITrainer):
    def __init__(self, cfg, model: IModel, train_loader: DataLoader, valid_loader: DataLoader, test_loader: DataLoader):
        super().__init__(cfg)

        self.model = model
        self.epoch_num = cfg.epoch_num
        self.seed = cfg.seed

        self.train_loader = train_loader
        self.valid_loader = valid_loader
        self.test_loader = test_loader

        self.fast_train = cfg.get('fast_train', False)
        self.fast_test = cfg.get('fast_test', False)
        self.process_bar = cfg.get('process_bar', True)

        if cfg.device=='auto':
            self.device = mk.get_current_device()
        else:
            self.device = torch.device(cfg.device if torch.cuda.is_available() else 'cpu')
            mk.set_current_device(cfg.device)
        self.model.set_trainer(self)
        log.info(f"Training Device: {self.device}")
        log.info(f"Training Epochs: {self.epoch_num}")

    def before_train(self):
        # 设置所有的随机数
        mk.seed_everything(self.seed)
        log.info(f"Seed Everything: {self.seed}, Strict Mode: False")
        # 初始化训练状态
        self.epoch_count = 0
        self.step_count = 0
        # 将模型传递至设备
        self.model.set_device(self.device)
        self.model.before_train()
        log.info('#' * 60)
        log.info('#' * 60)
        log.info('Start Training')

    ###################### 一个epoch的开始 ######################
    def before_epoch(self, epoch_id):
        log.info("\n")
        log.info(f"Epoch: {epoch_id} / {self.epoch_num}")
        self.model.before_epoch(epoch_id)
        # 进度条
        if self.process_bar:
            self.pbar = tqdm(total=len(self.train_loader)+len(self.valid_loader)+len(self.test_loader), desc=f"Epoch {epoch_id}")        

    ###################### 一个epoch的训练 ######################
    def train_epoch(self, epoch_id):
        mk.b_set('trainer_mode', 'train')
        self.model.train_mode()
        self.model.before_train_epoch(epoch_id)
        for batch_id, batch_data in enumerate(self.train_loader):
            # 在一个batch上训练数据
            batch_data = (lambda device=self.device, batch_data=batch_data: [k.to(device) for k in batch_data])()
            self.model.before_train_batch(epoch_id, batch_id, batch_data)
            self.model.train_batch(epoch_id, batch_id, batch_data)
            self.model.after_train_batch(epoch_id, batch_id, batch_data)
            if self.process_bar:
                self.pbar.update()
            self.step_count += 1
            if self.fast_train:
                break
        self.model.after_train_epoch(epoch_id)
        mk.b_set('trainer_mode', None)

    ###################### 一个epoch的验证 ######################
    def valid_epoch(self, epoch_id):
        mk.b_set('trainer_mode', 'valid')
        self.model.valid_mode()
        self.model.before_valid_epoch(epoch_id)
        for batch_id, batch_data in enumerate(self.valid_loader):
            # 在一个batch上验证数据
            batch_data = (lambda device=self.device, batch_data=batch_data: [k.to(device) for k in batch_data])()
            self.model.before_valid_batch(epoch_id, batch_id, batch_data)
            self.model.valid_batch(epoch_id, batch_id, batch_data)
            self.model.after_valid_batch(epoch_id, batch_id, batch_data)
            if self.process_bar:
                self.pbar.update()
            if self.fast_train:
                break
        self.model.after_valid_epoch(epoch_id)
        mk.b_set('trainer_mode', None)
    
    ###################### 一个epoch的测试 ######################
    def test_epoch(self, epoch_id):
        mk.b_set('trainer_mode', 'test')
        self.model.valid_mode()
        self.model.before_test_epoch(epoch_id)
        for batch_id, batch_data in enumerate(self.test_loader):
            # 在一个batch上测试数据
            batch_data = (lambda device=self.device, batch_data=batch_data: [k.to(device) for k in batch_data])()
            self.model.before_test_batch(epoch_id, batch_id, batch_data)
            self.model.test_batch(epoch_id, batch_id, batch_data)
            self.model.after_test_batch(epoch_id, batch_id, batch_data)
            if self.process_bar:
                self.pbar.update()
            if self.fast_test:
                break
        self.model.after_test_epoch(epoch_id)
        mk.b_set('trainer_mode', None)

    ###################### 一个epoch的结算 ######################
    def after_epoch(self, epoch_id):
        if self.process_bar:
            self.pbar.close()
        self.model.after_epoch(epoch_id)

    ###################### 训练被打断 ######################
    def on_train_interrupted(self):
        self.pbar.close()
        log.info('#' * 60)
        log.info('Early terminate')

    ###################### 训练结束 ######################
    def after_train(self):
        self.model.after_train()
        log.info('Training complete')

    ###################### 训练流程定义 ######################
    def train(self):
        self.before_train()
        try:
            for epoch_id in range(self.epoch_num):
                self.before_epoch(epoch_id)
                self.train_epoch(epoch_id)
                self.valid_epoch(epoch_id)
                self.test_epoch(epoch_id)
                self.after_epoch(epoch_id)
                self.epoch_count += 1
        except KeyboardInterrupt:
            self.on_train_interrupted()
        self.after_train()