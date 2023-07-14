# -*- coding: UTF-8 -*-

import os
import sys

import ipdb as pdb
import logging
log = logging.getLogger('BaseModel')

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
from torch.utils.data import DataLoader, Dataset
from .IModel import IModel
from mtools import monkey as mk
from functools import reduce
from basictorch_v3.Logger import Logger
try:
    from lightning.fabric.loggers import TensorBoardLogger
except:
    from lightning_fabric.loggers import TensorBoardLogger

import shutil

class BaseModel(IModel):
    def __init__(self, cfg, net, logger=None, save_on_metrics_name=["Valid Loss", "Test Error"], evaluate_on_metrics_names=["Valid Loss", "Test Error"], **kwargs):
        self.save_name = cfg.save_name
        self.epoch_stages_interval = getattr(cfg, 'epoch_stages_interval', -1)
        self.net = net
        self.logger = logger if logger is not None else Logger([TensorBoardLogger(root_dir='log', name='', version='')])
        self.last_lr = None
        self.auto_update_scheduler = kwargs.get('auto_update_scheduler', True)

        self.epoch_metrics_dict = {}
        self.history_metrics_dict = {}
        self.save_on_dict = {}
        self.evaluate_list = []

        if self.epoch_stages_interval > 0:
            log.info(f'Every {self.epoch_stages_interval} Epochs Change Model Output Dir')
        self.model_out_subpath = None

        num_params = sum(p.numel() for p in self.net.parameters() if p.requires_grad)
        log.info(f'Total number of parameters: {num_params}')

        self.optimizer = self.create_optimizer(cfg.optimizer, self.net)
        self.scheduler = self.create_scheduler(cfg.scheduler, self.optimizer)
        self.loss_function = self.create_loss_function(cfg.loss_function)
        log.info(f"{self.save_name=}")

        self.make_necessary_dirs()
        self.save_on_metrics(save_on_metrics_name)
        self.evaluate_on_metrics(evaluate_on_metrics_names)
    
    @property
    def device(self):
        if hasattr(self, 'trainer'):
            return self.trainer.device
        else:
            return mk.get_current_device()

    def make_necessary_dirs(self):
        os.makedirs("model", exist_ok=True)
        os.makedirs("figure", exist_ok=True)
        os.makedirs("evaluation", exist_ok=True)
        if self.epoch_stages_interval>0:
            self.model_out_subpath = f"epoch_num<{self.epoch_stages_interval}"
            os.makedirs(f"model/{self.model_out_subpath}", exist_ok=True)

    def set_device(self, device=None):
        self.net.to(mk.get_current_device() if device is None else device)

    def train_mode(self):
        self.net.train()

    def valid_mode(self):
        self.net.eval()

    def before_epoch(self, epoch_id):
        self.epoch_metrics_dict = {}

        if self.epoch_stages_interval>0:
            # 本epoch会执行目录切换
            if (epoch_id%self.epoch_stages_interval) == 0: 
                self.save("latest")
            
            # 获取当前这个epoch应该保存的目录
            cur_model_out_subpath = f"epoch_num<{((epoch_id//self.epoch_stages_interval)+1)*self.epoch_stages_interval}"

            if self.model_out_subpath != cur_model_out_subpath:
                shutil.copytree(f"model/{self.model_out_subpath}", f"model/{cur_model_out_subpath}", dirs_exist_ok=True)
                self.model_out_subpath = cur_model_out_subpath
            else:
                os.makedirs(f"model/{cur_model_out_subpath}", exist_ok=True)

    # 至少需要重载以下两个函数
    # 一个batch data的训练, 返回一个可以直接backward的损失值
    def train_step(self, epoch_id, batch_id, batch_data):
        return torch.tensor(0)
    
    # 一个batch data的测试, 返回一个CPU上的误差列表(np.array)
    def test_step(self, epoch_id, batch_id, batch_data):
        return np.array([0, 0, ...])
    
    # 一个batch data的验证, 返回一个可以直接backward的损失值, 但是valid_step不会进行backward
    def valid_step(self, epoch_id, batch_id, batch_data):
        return self.train_step(epoch_id, batch_id, batch_data)
    
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
        with torch.no_grad():
            error_list = self.test_step(epoch_id, batch_id, batch_data)
        if error_list is not None:
            # assert(isinstance(error_list, np.array))
            test_error_list = error_list[:, 0] if len(error_list.shape)>1 else error_list
            self.log_epoch_metrics('Test Error', test_error_list, reduce_fun=[np.hstack, np.mean])
       
    def after_epoch(self, epoch_id):
        # 输出各项指标
        self.log_history_metrics_dict(epoch_id)
        self.print_epoch_metrics(epoch_id)
        self.update_scheduler(epoch_id)
        self.save_model_on_epoch_end(epoch_id)

    def after_train(self):
        self.plot_losses()
        self.save("latest")
        self.save_epoch_metrics()
    
    def plot_losses(self):
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

    # 模型该如何在一个TestDataset上进行Evaluate, 返回一个包含各项指标的字典
    # out_dir指出了这个函数应该在什么地方保存自己的结果
    def evaluate_step(self, test_dataset, cfg, out_dir, model_name):
        if isinstance(test_dataset, DataLoader):
            test_loader = test_dataset
        elif isinstance(test_dataset, Dataset):
            test_loader = DataLoader(test_dataset, batch_size=cfg.batch_size, shuffle=False)
        else:
            raise TypeError(f"Unexpected test_dataset type :{type(test_dataset)}")

        for batch_id, batch_data in enumerate(test_loader):
            batch_data = mk.batch_to_device(batch_data)
            errors = self.test_step(0, batch_id, batch_data)
            mk.magic_append([errors.reshape(len(errors),-1)], "evaluate_batch_error")
        
        (error_list,) = mk.magic_get("evaluate_batch_error", np.vstack)
        error_cols = ['Err%d'%i for i in range(error_list.shape[1])]
        # err_df = pd.DataFrame({
        #     'Err': error_list,
        # })
        err_df = pd.DataFrame(error_list,columns=error_cols)

        des = err_df.describe(percentiles=[.25, .5, .75, .95, .99]).T

        err_df.to_csv(f"{out_dir}/eval_{model_name}_rst.csv", index=False)
        des.to_csv(f"{out_dir}/eval_{model_name}_des.csv")
        return {
            'Error Mean': np.around(error_list[:, 0].mean()*100, 3),
            'Error Std': np.around(error_list[:, 0].std()*100, 3),
        }
    
    def evaluate(self, test_dataset, cfg=None, suffix=None):
        device = mk.get_current_device()
        log.info("\n\n")
        log.info("Start Evaluating")
        log.info(f"Selecting Device: {device}")

        mk.write("Evaluation:")
        if self.epoch_stages_interval<0:
            for model_name in self.evaluate_list:
                model_path = f"model/{model_name}.pt"
                if os.path.exists(f"model/{model_name}.pt"):
                    self.load(model_path, load_opti=False)
                    self.set_device()
                    self.valid_mode()
                    out_dir = f"evaluation"
                    rst_dict = self.evaluate_step(test_dataset, cfg, out_dir, model_name)
                    mk.write(f"{model_name}:")
                    for key, val in rst_dict.items():
                        mk.write(f"    {key}: {val}")
        else:
            for sub_dir in sorted(os.listdir("model")):
                if not os.path.isdir(f"model/{sub_dir}"):
                    continue
                sub_dir_writed = False
                for model_name in self.evaluate_list:
                    model_path = f"model/{sub_dir}/{model_name}.pt"
                    if os.path.exists(model_path):
                        if not sub_dir_writed:
                            mk.write(f"{sub_dir}:")
                            sub_dir_writed = True
                        self.load(model_path, load_opti=False)
                        self.set_device()
                        self.valid_mode()
                        out_dir = f"evaluation/{sub_dir}"
                        os.makedirs(out_dir, exist_ok=True)
                        rst_dict = self.evaluate_step(test_dataset, cfg, out_dir, model_name)
                        mk.write(f"    {model_name}:")
                        for key, val in rst_dict.items():
                            mk.write(f"        {key}: {val}")

        mk.save("Evaluation.log")
        with open("Evaluation.log", 'r', encoding='utf-8') as f:
            log.info(f.read())
                    
        print('tensorboard --logdir=%s'%os.path.realpath(""))

    def save(self, file_name="latest"):
        if self.model_out_subpath:
            save_path = f"model/{self.model_out_subpath}/{file_name}.pt"
        else:
            save_path = f"model/{file_name}.pt"
        torch.save({
            'net_state_dict': self.net.state_dict(),
            'optimizer_state_dict': self.optimizer.state_dict(),
            'scheduler_state_dict': self.scheduler.state_dict(),
        }, save_path)
    
        log.info(f'Model saved to {save_path}')        

    def load(self, model_path, device=None, strict=True, load_opti=True):
        checkpoint = torch.load(model_path, map_location=mk.get_current_device() if device is None else device)
        self.net.load_state_dict(checkpoint['net_state_dict'], strict=strict)
        if load_opti:
            self.optimizer.load_state_dict(checkpoint['optimizer_state_dict'])
            self.scheduler.load_state_dict(checkpoint['scheduler_state_dict'])
    

    ##############################################Tools##################################################
    def get_optimizer_parameters(self, net, cfg=None):
        return net.parameters() if hasattr(net, 'parameters') else net

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

    def log_epoch_metrics(self, name, value, reduce_fun=np.mean, save_name=None, save_group=None):
        if reduce_fun is None:
            raise RuntimeError(f"epoch_metrics Must Have A Reduce Function")
        
        if value is None:
            return

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
            self.log_epoch_metrics(name, metrics_dict[name], reduce_fun=reduce_fun, save_name=save_name, save_group=save_group)
    
    def log_history_metrics_dict(self, epoch_id):
        metrics_groups = {'':{}}
        for name in self.epoch_metrics_dict.keys():
            reduce_fun = self.epoch_metrics_dict[name]['reduce']
            if isinstance(reduce_fun, list):
                try:
                    reduce_result = reduce(lambda t,f: f(t), reduce_fun, self.epoch_metrics_dict[name]['values'])
                except:
                    pdb.set_trace()
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
        max_name_len = np.max([len(_) for _ in self.history_metrics_dict.keys()])
        for name in sorted(self.history_metrics_dict.keys()):
            if self.history_metrics_dict[name]['epoch_id'][-1]==epoch_id:
                log.info('%*s : %.6f'%(int(max_name_len), name, self.history_metrics_dict[name]['values'][-1]))

    def update_scheduler(self, epoch_id, monitor="Valid Loss"):
        # Scheduler 更新
        if self.auto_update_scheduler:
            if type(self.scheduler) == torch.optim.lr_scheduler.ReduceLROnPlateau:
                valid_loss_mean = self.get_epoch_metrics(monitor)
                self.scheduler.step(valid_loss_mean)
            else:
                self.scheduler.step()
            if self.last_lr is None:
                self.last_lr = self.scheduler._last_lr[0]
            if self.last_lr != self.scheduler._last_lr[0]:
                log.info(f"Updating LR: {self.last_lr}->{self.scheduler._last_lr[0]}")
                self.last_lr = self.scheduler._last_lr[0]
    
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
            index_list, val_list = self.get_epoch_metrics_history(name)
            save_name = self.history_metrics_dict[name]['save_name']
            tmp_df = pd.DataFrame(index=index_list, data={
                save_name:val_list
            })
            df = pd.concat([df, tmp_df], axis=1)

        df.reset_index(inplace=True, drop=False)
        df.rename(columns={"index": "epoch_id"}, inplace=True)
        df.to_csv("figure/epoch_metrics.csv", index=False, header=True)

    # 一个epoch过后网络的保存
    def save_model_on_epoch_end(self, epoch_id):
        for metrics_name in self.history_metrics_dict.keys():
            if metrics_name in self.save_on_dict:
                old_val = self.save_on_dict[metrics_name]['best_val']
                now_val = self.history_metrics_dict[metrics_name]['values'][-1]
                save_name = self.save_on_dict[metrics_name]['save_name']
                if self.save_on_dict[metrics_name]['mode']=='min' and now_val < old_val:
                    self.save_on_dict[metrics_name]['best_val'] = now_val
                    self.save(save_name)
                elif self.save_on_dict[metrics_name]['mode']=='max' and now_val > old_val:
                    self.save_on_dict[metrics_name]['best_val'] = now_val
                    self.save(save_name)
    
    def save_on_metrics(self, metrics_name, save_name=None, mode='min'):
        if isinstance(metrics_name, list):
            for _ in metrics_name:
                self.save_on_metrics(_, None, mode)
            return

        if metrics_name in self.save_on_dict:
            log.warn(f"{metrics_name} Already Registered, Re Registering")

        self.save_on_dict[metrics_name] = {
            'save_name': save_name if save_name is not None else f"{metrics_name.replace(' ', '_')}_best",
            'mode': mode,
            'best_val': np.inf if mode=='min' else -np.inf,
        }
    
    def evaluate_on_metrics(self, metrics_name):
        if isinstance(metrics_name, list):
            for _ in metrics_name:
                self.evaluate_on_metrics(_)
            return
        self.evaluate_list.append(f"{metrics_name.replace(' ', '_')}_best")