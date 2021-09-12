import torch
import torch.nn as nn
import torch.optim as optim
import numpy as np
import basictorch.tools as t
from mtools import list_ind,tuple_ind
from .losses import loss_funcs

class Base(nn.Module):
    def __init__(self, name, args, set_params=True, **model_params):
        super().__init__()
        self.name = name
        self.args = args # require args.output, args.data_name, args.data_ver, args.exp_no
        self.loss_funcs = {}
        self.default_params = {}
        self.args_params = []
        if set_params:
            self.set_params(model_params)
    
    def add_default_params(self, default_params):
        self.default_params = t.merge_params(self.default_params, default_params)
    
    def add_args_params(self, args_params):
        self.args_params = list(set(self.args_params + args_params))
    
    # def set_params(self, model_params):
        # self.add_default_params(default_params)
        # self.add_args_params(args_params)
        # super().set_params(self, model_params)

    # priority: model_params > args_params > default_model_params
    def set_params(self, model_params):
        self.model_params = model_params
        for param in self.args_params:
            if (param not in self.model_params) and hasattr(self.args, param):
                self.model_params[param] = t.get_param(self.args.__dict__, param)
        self.model_params = t.merge_params(self.model_params, self.default_params)
        print('%s model_params: %s' % (self.name, str(self.model_params)))
        for param in self.model_params:
            self.__dict__[param] = t.get_param(self.model_params, param)
        self.build_model()
    
    def build_model(self):
        self.sequential = nn.Sequential()

    def forward(self, x):
        return self.sequential(x)

    def train_mode(self, mode):
        super().train(mode)

    def train(self, batch_size=0, epochs=0, validation=True, reporters=[], monitor=None, test_monitor=None, initialize=True, max_sub_size=1e3):
        if type(batch_size) != bool:
            self.batch_size = batch_size
            self.epochs = epochs
            if hasattr(self.args, 'test_model'):
                if self.args.test_model:
                    self.epochs = 1
            self.validation = validation
            self.reporters = reporters
            self.monitor = monitor if monitor else (self.monitor if hasattr(self, 'monitor') else 'loss')
            self.test_monitor = test_monitor if test_monitor else self.monitor
            self.initialize = initialize
            self.max_sub_size = max_sub_size
            if self.epochs>0:
                self.on_train_begin()
                for e in range(self.epochs):
                    self.on_epoch_begin(e)
                    self.train_on_epoch()
                    self.on_epoch_end()
                self.on_train_end()
            else:
                self.val_epoch = 0
                self.fit_time = 0
        else:
            mode = batch_size
            self.train_mode(mode)
    
    def on_train_begin(self):
        print('\n\n--- START TRAINING ---\n\n')
        t.set_weight_file(self)
        self.history = {}
        self.train_fit_time = t.time.process_time()
        self.b = 0
        self.epoch = -1
        if self.initialize:
            self.initialize_model()
        self.check_validation()
    
    def on_train_end(self):
        self.fit_time = t.time.process_time() - self.train_fit_time
        if self.validation:
            self.load_state_dict(torch.load(self.weights_file))
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()
        print('\n\n--- FINISH TRAINING ---\n\n')
        self.save_end()

    def on_epoch_begin(self, epoch):
        self.train_mode(True)
        self.epoch_start_time = t.time.process_time()
        self.epoch = epoch
        self.data_loader = self.datasets.get_train_loader()
    
    def train_on_epoch(self):
        for b, batch_data in enumerate(self.data_loader):
            self.b = b
            losses = self.train_on_batch(b, batch_data)
            self.print_batch(b, losses)
    
    def print_batch(self, b, losses):
        if hasattr(self.args, 'print_batch'):
            if self.args.print_batch:
                t.print_batch(self.epoch, self.epochs, b, self.batch_size, self.num_data, losses)
            else:
                t.sys.stdout.flush()
        else:
            t.print_batch(self.epoch, self.epochs, b, self.batch_size, self.num_data, losses)

    def on_epoch_end(self):
        epoch_time = t.time.process_time() - self.epoch_start_time
        losses = self.evaluate()
        self.check_validation(losses)
        self.add_losses_to_history(losses)
        t.print_epoch(self.epoch, self.epochs, losses, epoch_time)

    def train_on_batch(self, b, batch_data):
        self.optimizer.zero_grad()
        losses = self.get_losses(batch_data)
        losses['loss'].backward()
        self.optimizer.step()
        return losses

    def unpack_batch(self, batch_data, batch_i=None, default_batch_i=[0,1]):
        # batch_i > self.batch_i > default_batch_i
        if batch_i is None:
            batch_i = self.batch_i if hasattr(self, 'batch_i') else default_batch_i
        return tuple_ind(batch_data, batch_i)

    def get_losses(self, batch_data, batch_i=None):
        inputs, labels = self.unpack_batch(batch_data, batch_i)
        outputs = self(inputs)
        losses={}
        for r in self.loss_funcs:
            losses[r] = self.loss_funcs[r](outputs, labels)
        return losses

    def get_dataset_losses(self, dataset, eval_dataset='train'):
        batch_data=tuple(dataset.tensors)
        self.eval_dataset = eval_dataset
        return self.get_batchdata_losses(batch_data)
        
    def get_batchdata_losses(self, batch_data):
        with torch.no_grad():
            self.train_mode(False)
            losses_list = []
            for sub_batch_data in t.get_sub_batch_data(batch_data, self.max_sub_size):
                losses_list.append(self.get_losses(sub_batch_data))
            losses = t.merge_losses(losses_list)
            return losses

    def set_datasets(self, datasets):
        self.datasets = datasets
        self.num_data = len(datasets.train_dataset)

    def initialize_model(self):
        t.reset_parameters(self)
        t.initialize_model(self)

    def evaluate(self):
        losses = self.get_dataset_losses(self.datasets.train_dataset)
        if hasattr(self.datasets, 'test_dataset'):
            test_losses = self.get_dataset_losses(self.datasets.test_dataset, eval_dataset='test')
            losses['test_'+self.test_monitor] = test_losses[self.test_monitor]
            for r in test_losses:
                if 'test_'+r in self.reporters:
                    losses['test_'+r] = test_losses[r]
        if self.validation:
            val_losses = self.get_dataset_losses(self.datasets.val_dataset, eval_dataset='val')
            losses['val_'+self.monitor] = val_losses[self.monitor]
        return losses

    def check_validation(self, losses=None):
        if self.validation:
            if not losses:
                self.on_epoch_begin(-1)
                val_losses = self.get_dataset_losses(self.datasets.val_dataset)
                if self.monitor in val_losses:
                    self.monitor_loss = val_losses[self.monitor]+1
                    self.on_epoch_end()
                self.val_epoch = self.epoch
            else:
                if (losses['val_'+self.monitor] < self.monitor_loss) and not torch.isnan(losses['val_'+self.monitor]):
                    self.monitor_loss = losses['val_'+self.monitor]
                    self.monitor_losses = losses
                    torch.save(self.state_dict(), self.weights_file)
                    self.val_epoch = self.epoch
        else:
            self.monitor_losses = losses

    def add_losses_to_history(self, losses):
        for r in list(losses.keys()):
            if r in self.history:
                self.history[r].append(losses[r].item())
            else:
                self.history[r] = [losses[r].item()]

    def save_end(self):
        self.save_model()
        self.save_curve()
        self.save_evaluate()

    def save_model(self):
        t.save_model(self)
    
    def load_model(self):
        t.load_model(self)
    
    def save_curve(self):
        t.curve_plot(self.history, self.args, curve_name='curve_%s' % self.name)

    def save_evaluate(self, postfix=''):
        if self.validation:
            head = 'data_ver,data_name,exp_no,epochs,batch_size,%s,%s,fit_time,val_epoch\n' % (self.monitor, 'test_%s'%self.test_monitor)
            varList = [self.args.data_ver, self.args.data_name, self.args.exp_no, self.epochs,self.batch_size,\
                self.monitor_losses[self.monitor].item(), self.monitor_losses['test_%s'%self.test_monitor].item() if 'test_%s'%self.test_monitor in self.monitor_losses else 'None', self.fit_time, self.val_epoch]
        else:
            head = 'data_ver,data_name,exp_no,epochs,batch_size,loss,fit_time\n'
            varList = [self.args.data_ver, self.args.data_name, self.args.exp_no, self.epochs,self.batch_size,\
                self.history['loss'][-1], self.fit_time]
        if hasattr(self.args, 'run_i'):
            head = 'run_i,'+head
            varList.insert(0, self.args.run_i)
        t.save_evaluate(self.args.output, '%s%s'%(self.name,postfix), head, varList)
    
    def apply_func(self, func=None, func_params={}):
        if func:
            func(self, **func_params)
    
    # abandoned function, use basictorch.tools.get_model_params instead
    def get_model_params(self, model_params, _default_model_params):
        for param in _default_model_params:
            if param not in model_params:
                model_params[param] = _default_model_params[param]
        return model_params

class SemiBase(Base):
    def on_epoch_begin(self, epoch):
        super().on_epoch_begin(epoch)
        self.data_loader = self.datasets.get_train_loader()
        self.unlab_loader = self.datasets.get_unlab_loader(len(self.data_loader))
    
    def on_epoch_end(self):
        super().on_epoch_end()
        # if torch.cuda.is_available():
        #     torch.cuda.empty_cache()

    def train_on_epoch(self):
        for (b, batch_data_l),(b, batch_data_u) in zip(enumerate(self.data_loader), enumerate(self.unlab_loader)):
            batch_data = batch_data_l+batch_data_u
            losses = self.train_on_batch(b, batch_data)
            self.print_batch(b, losses)
    
    def get_dataset_losses(self, dataset, eval_dataset='train'):
        batch_data = tuple(dataset.tensors)+tuple(self.datasets.get_unlab_dataset(dataset).tensors)
        self.eval_dataset = eval_dataset
        return self.get_batchdata_losses(batch_data)
    
    def get_losses(self, batch_data, batch_i = None):
        inputs, labels, unlabs = self.unpack_batch(batch_data, batch_i, [0,1,2])
        outputs = self(inputs, unlabs)
        losses={}
        for r in self.loss_funcs:
            losses[r] = self.loss_funcs[r](outputs, labels)
        return losses
    
    def forward(self, x, x_u):
        return self.sequential(x)

acts = {
    'relu':torch.relu,
    'tanh':torch.tanh,
    'sigmoid':torch.sigmoid,
    'leakyrelu':torch.nn.functional.leaky_relu,
    'elu':torch.nn.functional.elu,
    'softmax':torch.softmax,
}

act_modules = {
    'relu':nn.ReLU(),
    'tanh':nn.Tanh(),
    'sigmoid':nn.Sigmoid(),
    'leakyrelu':torch.nn.modules.LeakyReLU(),
    'elu':nn.ELU(),
    'softmax':torch.nn.Softmax(),
}

poolings={
    'max':nn.MaxPool2d(2),
    'avg':nn.AvgPool2d(2),
}