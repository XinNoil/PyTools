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
from functools import reduce
from basictorch_v3.Logger import Logger
from lightning.fabric.loggers import TensorBoardLogger

class BaseModel(IModel):
    def __init__(self, cfg, net, logger=None, **kwargs):
        self.save_name = cfg.save_name
        self.net = net
        self.logger = logger if logger is not None else Logger([TensorBoardLogger(root_dir='log', name='', version='')])
        self.last_lr = None
        self.auto_update_scheduler = kwargs.get('auto_update_scheduler', True)
        self.auto_save_model_on_epoch_end = kwargs.get('auto_save_model_on_epoch_end', True)

        self.epoch_metrics_dict = {}
        self.history_metrics_dict = {}

        num_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        log.info(f'Total number of parameters: {num_params}')

        self.optimizer = self.create_optimizer(cfg.optimizer, self.net)
        self.scheduler = self.create_scheduler(cfg.scheduler, self.optimizer)
        self.loss_function = self.create_loss_function(cfg.loss_function)
        log.info(f"{self.save_name=}")

        self.make_necessary_dirs()

    def make_necessary_dirs(self):
        os.makedirs("model", exist_ok=True)
        os.makedirs("figure", exist_ok=True)

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

    def before_epoch(self, epoch_id):
        self.epoch_metrics_dict = {}
        
    # 至少需要重载以下两个函数
    # 一个batch data的训练, 返回一个可以直接backward的损失值
    def train_step(self, epoch_id, batch_id, batch_data):
        return torch.tensor(0)
    # 一个batch data的测试, 返回一个CPU上的误差列表(np.array)
    def test_step(self, epoch_id, batch_id, batch_data):
        return torch.tensor(0)
    
    # 一个batch data的验证, 返回一个可以直接backward的损失值, 但是valid_step不会进行backward
    def valid_step(self, epoch_id, batch_id, batch_data):
        return self.train_step(epoch_id, batch_id, batch_data)
    # 一个batch data的测试, 返回一个CPU上的误差列表(np.array)
    def evaluate_step(self, batch_id, batch_data):
        return self.test_step(0, batch_id, batch_data)

    def train_batch(self, epoch_id, batch_id, batch_data):
        self.optimizer.zero_grad()
        loss = self.train_step(epoch_id, batch_id, batch_data)
        self.log_epoch_metrics('Train Loss', loss.item())
        loss.backward()
        self.optimizer.step()

    def valid_batch(self, epoch_id, batch_id, batch_data):
        loss = self.valid_step(epoch_id, batch_id, batch_data)
        if loss is not None:
            self.log_epoch_metrics('Valid Loss', loss.item())

    def test_batch(self, epoch_id, batch_id, batch_data):
        error_list = self.test_step(epoch_id, batch_id, batch_data)
        if error_list is not None:
            # assert(isinstance(error_list, np.array))
            self.log_epoch_metrics('Test Error', error_list, reduce_fun=[np.hstack, np.mean])
       
    def after_epoch(self, epoch_id):
        # 输出各项指标
        self.log_history_metrics_dict(epoch_id)
        self.print_epoch_metrics(epoch_id)
        self.update_scheduler(epoch_id)
        self.save_model_on_epoch_end(epoch_id)

    def update_scheduler(self, epoch_id):
        # Scheduler 更新
        if self.auto_update_scheduler:
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                valid_loss_mean = self.get_epoch_metrics("Valid Loss")
                self.scheduler.step(valid_loss_mean)
            else:
                self.scheduler.step()
            if self.last_lr is None:
                self.last_lr = self.scheduler._last_lr[0]
            if self.last_lr != self.scheduler._last_lr[0]:
                log.info(f"Updating LR: {self.last_lr}->{self.scheduler._last_lr[0]}")
                self.last_lr = self.scheduler._last_lr[0]
        
    def save_model_on_epoch_end(self, epoch_id):
        # 一个epoch过后网络的保存
        if self.auto_save_model_on_epoch_end:
            valid_loss_mean = self.get_epoch_metrics("Valid Loss")
            # 保存验证集误差最小的模型
            if valid_loss_mean and valid_loss_mean < self.best_val_loss:
                self.best_val_loss = valid_loss_mean
                self.save("valbest")

            # 保存测试集误差最小的模型
            err_mean = self.get_epoch_metrics("Test Error")
            if err_mean and err_mean < self.best_test_loss:
                self.best_test_loss = err_mean
                self.save("testbest")

    def after_train(self):
        # 位移预测任务 训练集\验证集\测试集 损失\误差
        fig, ax = plt.subplots(figsize=(8, 6))
        ax.plot(self.history_metrics_dict["Train Loss"]['values'], label="Train Loss")
        if "Valid Loss" in self.history_metrics_dict:
            ax.plot(self.history_metrics_dict["Valid Loss"]['values'], label="Valid Loss")
        if "Test Error" in self.history_metrics_dict:
            ax.plot(self.history_metrics_dict["Test Error"]['values'], label="Test Error")
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
        
        print('tensorboard --logdir=%s'%os.path.realpath(""))


    def save(self, message="latest"):
        save_path = f"model/{self.save_name}_{message}.pt"
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)
    
        log.info(f'Model saved to {save_path}')        

    def load(self, model_path):
        checkpoint = torch.load(model_path, map_location='cuda:0')
        self.net.load_state_dict(checkpoint['net_state_dict'])
        self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
        self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    

    ##############################################Tools##################################################
    def get_optimizer_parameters(self, net, cfg=None):
        return net.parameters()

    def create_optimizer(self, cfg, net):
        optimizer_type = cfg.type
        optimizer_params = mk.eval_dict_values(cfg.params)
        optimizer_class = mk.get_class("torch.optim", optimizer_type)
        optimizer = optimizer_class(self.get_optimizer_parameters(net, cfg), **optimizer_params)
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

    # def refresh_epoch_metrics(self):
    #     for name in self.epoch_metrics_dict.keys():
    #         self.epoch_metrics_dict[name]['values'] = []
    #         self.epoch_metrics_dict[name]['is_reduced'] = False

    def log_epoch_metrics(self, name, value, reduce_fun=np.mean, save_name=None, save_group=None):
        if reduce_fun is None:
            raise RuntimeError(f"epoch_metrics Must Have A Reduce Function")

        if name not in self.epoch_metrics_dict:
            self.epoch_metrics_dict[name] = {
                'values': [],
                'reduce': reduce_fun,
                'save_name': save_name if save_name is not None else name.replace(" ", "_"),
                'save_group': save_group if save_group is not None else ''
            }
        if isinstance(value, torch.Tensor):
            value = value.item()
            
        self.epoch_metrics_dict[name]['values'].append(value)
    
    def log_epoch_metrics_dict(self, metrics_dict, reduce_fun=np.mean, save_name=None, save_group=None):
        if reduce_fun is None:
            raise RuntimeError(f"epoch_metrics Must Have A Reduce Function")
        for name in metrics_dict:
            self.log_epoch_metrics(name, metrics_dict[name], reduce_fun=np.mean, save_name=save_name, save_group=save_group)
    
    def log_history_metrics_dict(self, epoch_id):
        metrics_groups = {'':{}}
        for name in self.epoch_metrics_dict.keys():
            reduce_fun = self.epoch_metrics_dict[name]['reduce']
            if isinstance(reduce_fun, list):
                reduce_result = reduce(lambda t,f: f(t), reduce_fun, self.epoch_metrics_dict[name]['values'])
            else:
                reduce_result = reduce_fun(self.epoch_metrics_dict[name]['values'])
            if name not in self.history_metrics_dict:
                self.history_metrics_dict[name] = {
                    'values': [],
                    'epoch_id': [],
                    'save_name': self.epoch_metrics_dict[name]['save_name']
                }
            self.history_metrics_dict[name]['values'].append(reduce_result)
            self.history_metrics_dict[name]['epoch_id'].append(epoch_id)
            save_group = self.epoch_metrics_dict[name]['save_group']
            if save_group in metrics_groups:
                metrics_groups[save_group][name] = reduce_result
            else:
                metrics_groups[save_group] = {name:reduce_result}
        for metrics_group in metrics_groups:
            self.logger.log(metrics_group, metrics_groups[metrics_group], epoch_id)
    
    def print_epoch_metrics(self, epoch_id):
        for name in self.history_metrics_dict.keys():
            if self.history_metrics_dict[name]['epoch_id'][-1]==epoch_id:
                log.info(f"{name}: {self.history_metrics_dict[name]['values'][-1]:.6f}")

    def get_epoch_metrics(self, name, epoch_id=-1):
        if name not in self.history_metrics_dict:
            log.error(f"Trying To Get epoch_metrics: {name}, Not Found, Returning None")
            return None
        return self.history_metrics_dict[name]['values'][epoch_id]

    def get_epoch_metrics_history(self, name):
        if name not in self.history_metrics_dict:
            log.error(f"Trying To Get epoch_metrics: {name}, Not Found, Returning None")
            return None
        return self.history_metrics_dict[name]['epoch_id'], self.history_metrics_dict[name]['values']
    
    def save_epoch_metrics(self):
        df = pd.DataFrame()

        for name in self.history_metrics_dict.keys():
            history_list = self.history_metrics_dict[name]['values']
            save_name = self.history_metrics_dict[name]['save_name']
            df[save_name] = history_list

        df.to_csv("figure/epoch_metrics.csv", index=False, header=True)