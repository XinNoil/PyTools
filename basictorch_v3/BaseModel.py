# -*- coding: UTF-8 -*-

import os
import sys

import ipdb as pdb
import logging
log = logging.getLogger(__name__)

import numpy as np
np.set_printoptions(precision=4, suppress=True, formatter={'float_kind':'{:f}'.format})

import pandas as pd
pd.set_option('display.float_format','{:.4f}'.format)

import matplotlib.pyplot as plt
import matplotlib
matplotlib.rcParams['font.family'] = ['Times New Roman']
matplotlib.rcParams['axes.unicode_minus'] = False
matplotlib.rcParams['axes.titlesize'] = 20
matplotlib.rcParams['axes.labelsize'] = 18
matplotlib.rcParams['figure.titlesize'] = 24
matplotlib.rcParams['xtick.labelsize'] = 14
matplotlib.rcParams['ytick.labelsize'] = 14
matplotlib.rcParams['legend.fontsize'] = 16

import torch
from .IModel import IModel
from mtools import monkey as mk

class BaseModel(IModel):
    def __init__(self, cfg, net):
        self.save_name = cfg.save_name
        self.net = net
        self.last_lr = None
        self.auto_update_scheduler = True
        self.auto_save_model_on_epoch_end = True

        num_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        log.info(f'Total number of parameters: {num_params}')

        self.optimizer = self.create_optimizer(cfg.optimizer, self.net)
        self.scheduler = self.create_scheduler(cfg.scheduler, self.optimizer)
        self.loss_function = self.create_loss_function(cfg.loss_function)

    def set_device(self, device):
        self.net.to(device)

    def train_mode(self):
        self.net.train()

    def valid_mode(self):
        self.net.eval()

    def before_train(self):
        # 当前最好的验证集和测试机损失是多少, 初始化为无穷
        self.best_val_loss = np.inf
        self.best_test_loss = np.inf


    # 至少需要重载以下两个函数
    # 一个batch data的训练, 返回一个可以直接backward的损失值
    def train_step(self, epoch_id, batch_id, batch_data):
        return {'loss':0}
    # 一个batch data的测试, 返回一个CPU上的误差列表(np.array)
    def test_step(self, epoch_id, batch_id, batch_data):
        return {'loss':0}
    
    # 一个batch data的验证, 返回一个可以直接backward的损失值, 但是valid_step不会进行backward
    def valid_step(self, epoch_id, batch_id, batch_data):
        return self.train_step(epoch_id, batch_id, batch_data)
    # 一个batch data的测试, 返回一个CPU上的误差列表(np.array)
    def evaluate_step(self, batch_id, batch_data):
        return self.test_step(0, batch_id, batch_data)

    def train_batch(self, epoch_id, batch_id, batch_data):
        self.optimizer.zero_grad()
        losses = self.train_step(epoch_id, batch_id, batch_data)
        mk.magic_append([losses[_].item() for _ in losses], "train_batch_loss")
        losses['loss'].backward()
        self.optimizer.step()

    def valid_batch(self, epoch_id, batch_id, batch_data):
        losses = self.valid_step(epoch_id, batch_id, batch_data)
        mk.magic_append([losses[_].item() for _ in losses], "valid_batch_loss")  

    def test_batch(self, epoch_id, batch_id, batch_data):
        losses = self.test_step(epoch_id, batch_id, batch_data)
        mk.magic_append([losses[_].item() for _ in losses], "test_batch_error")

    def after_epoch(self, epoch_id):
        # 这一个 epoch 的 训练集 验证集 测试集误差
        train_batch_loss = mk.magic_get("train_batch_loss", np.mean)
        valid_batch_loss = mk.magic_get("valid_batch_loss", np.mean)
        mk.magic_append(train_batch_loss, "train_epoch_loss")
        mk.magic_append(valid_batch_loss, "valid_epoch_loss")
        
        log.info(f"Avg Train Loss: {train_batch_loss[0]:.6f}")
        log.info(f"Avg Valid Loss: {valid_batch_loss[0]:.6f}")
        
        # 这一个 epoch 的测试集误差, 注意这里是每个样本的, 不是每个batch平均后的
        [err_all] = mk.magic_get("test_batch_error", np.hstack)
        err_mean = np.mean(err_all)
        mk.magic_append([err_mean], "test_epoch_error")
        log.info(f"Test Error: {err_mean:.6f}")

        # Scheduler 更新
        if self.auto_update_scheduler:
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                self.scheduler.step(valid_batch_loss[0])
            else:
                self.scheduler.step()
            if self.last_lr is None:
                    self.last_lr = self.scheduler._last_lr[0]
            if self.last_lr != self.scheduler._last_lr[0]:
                log.info(f"Updating LR: {self.last_lr}->{self.scheduler._last_lr[0]}")
                self.last_lr = self.scheduler._last_lr[0]
    
        # 一个epoch过后网络的保存
        if self.auto_save_model_on_epoch_end:
            # 保存验证集误差最小的模型
            if valid_batch_loss[0] < self.best_val_loss:
                self.best_val_loss = valid_batch_loss[0]
                self.save("valbest")

            # 保存测试集误差最小的模型
            if err_mean < self.best_test_loss:
                self.best_test_loss = err_mean
                self.save("testbest")

        return train_batch_loss, valid_batch_loss, err_mean

    
    def after_train(self):
        train_losses_all = mk.magic_get("train_epoch_loss")
        valid_losses_all = mk.magic_get("valid_epoch_loss")
        [test_error_all] = mk.magic_get("test_epoch_error")
        df = pd.DataFrame({
            'train_loss': train_losses_all[0],
            'valid_loss': valid_losses_all[0],
            'test_error': test_error_all,
        })

        os.makedirs("figure", exist_ok=True)
        df.to_csv("figure/losses.csv", index=False, header=True)

        # 位移预测任务 训练集\验证集\测试集 损失\误差
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(train_losses_all[0], label="Train Loss")
        ax.plot(valid_losses_all[0], label="Valid Loss")
        ax.plot(test_error_all, label="Test Error")
        ax.set_xlabel("Epoch")
        ax.set_ylabel("Loss or Error")
        ax.legend()
        fig.tight_layout()
        fig.savefig(f"figure/losses.png", dpi=300)
        plt.close(fig)

    def evaluate(self, test_loader, suffix=None):
        gpu_id = mk.get_free_gpu()
        device = f"cuda:{gpu_id}"
        log.info(f"Auto Selecting cuda:{gpu_id} GPU")
        self.set_device(device)
        self.valid_mode()

        for batch_id, batch_data in enumerate(test_loader):
            batch_data = (lambda device=device, batch_data=batch_data: [k.to(device) for k in batch_data])()
            error = self.evaluate_step(batch_id, batch_data)
            mk.magic_append([error], "evaluate_batch_error")

        [err] = mk.magic_get("evaluate_batch_error", np.hstack)

        err_df = pd.DataFrame({
            'Err': err,
        })

        des = err_df.describe(percentiles=[.25, .5, .75, .95, .99]).T
        log.info(f"Evaluate Results:\n{des}")

        if suffix:
            err_df.to_csv(f"eval_rst_{suffix}.csv", index=False)
            des.to_csv(f"eval_des_{suffix}.csv")
        else:
            err_df.to_csv(f"eval_rst.csv", index=False)
            des.to_csv(f"eval_des.csv")


    def save(self, message="latest"):
        os.makedirs("model", exist_ok=True)
        save_path = f"model/{self.save_name}_{message}.pt"
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
        }, save_path)
    
        log.info(f'Model saved to {save_path}')        

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location='cuda:0')
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
    

    ##############################################Tools##################################################
    def get_optimizer_parameters(self, net):
        return net.parameters()

    def create_optimizer(self, cfg, net):
        optimizer_type = cfg.type
        optimizer_params = mk.eval_dict_values(cfg.params)
        optimizer_class = mk.get_class("torch.optim", optimizer_type)
        optimizer = optimizer_class(self.get_optimizer_parameters(net), **optimizer_params)
        log.info(f"Created Optimizer: {optimizer_class} With {optimizer_params}")
        return optimizer
        
    def create_scheduler(self, cfg, optimizer):
        scheduler_type = cfg.type
        scheduler_params = mk.eval_dict_values(cfg.params)
        scheduler_class = mk.get_class("torch.optim.lr_scheduler", scheduler_type)
        scheduler = scheduler_class(optimizer, **scheduler_params)
        log.info(f"Created Scheduler: {scheduler_class} With {scheduler_params}")
        return scheduler

    def create_loss_function(self, cfg):
        loss_function_type = cfg.type
        loss_function_params = mk.eval_dict_values(cfg.params)
        loss_function_class = mk.get_class("torch.nn", loss_function_type)
        loss_function = loss_function_class(**loss_function_params)
        log.info(f"Loss Function: {loss_function_class} With {loss_function_params}")
        return loss_function