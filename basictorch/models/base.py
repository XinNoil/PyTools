import torch,os,copy
import torch.nn as nn
import torch.optim as optim
import numpy as np
import basictorch.tools as t
from .losses import loss_funcs

class Base(nn.Module): #, metaclass=abc.ABCMeta
    def __init__(self, name, args, set_model_params=True, **model_params):
        super().__init__()
        self.name = name
        self.args = args # args.output, self.args.data_name, args.data_ver, self.args.exp_no
        self.loss_funcs = {}
        if set_model_params:
            self.set_model_params(model_params)
    
    def train_mode(self, mode):
        super().train(mode)

    def set_model_params(self, model_params, default_model_params={}):
        self.model_params = self.get_model_params(model_params, default_model_params)
        for param in model_params:
            self.__dict__[param] = model_params[param].copy() if isinstance(model_params[param], list) else model_params[param]
    
    def get_model_params(self, model_params, default_model_params):
        for param in default_model_params:
            if param not in model_params:
                model_params[param] = default_model_params[param]
        return model_params
            
    def build_model(self):
        self.sequential = nn.Sequential()

    def forward(self, x):
        return self.sequential(x)

    def train(self, batch_size=16, epochs=0, validation=True, reporters=['loss','test_loss'], monitor='loss', test_monitor=None, initialize=True):
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation = validation
        self.reporters = reporters
        self.monitor = monitor
        self.test_monitor = test_monitor if test_monitor else 'test_%s' % self.monitor
        self.initialize = initialize
        if epochs>0:
            self.on_train_begin()
            for e in range(self.epochs):
                self.on_epoch_begin(e)
                self.train_on_epoch()
                self.on_epoch_end()
            self.on_train_end()
        else:
            self.fit_time = 0
    
    def on_train_begin(self):
        print('\n\n--- START TRAINING ---\n\n')
        t.set_weight_file(self)
        self.history = {}
        self.train_fit_time = t.time.process_time()
        if self.initialize:
            self.initialize_model()
        self.check_validation()
    
    def on_train_end(self):
        self.fit_time = t.time.process_time() - self.train_fit_time
        if self.validation:
            self.load_state_dict(torch.load(self.weights_file))
        if torch.cuda.is_available():
            torch.cuda.empty_cache()
        print('\n\n--- FINISH TRAINING ---\n\n')
        self.save_end()

    def on_epoch_begin(self, epoch):
        self.train_mode(True)
        self.epoch_start_time = t.time.process_time()
        self.epoch = epoch
    
    def train_on_epoch(self):
        data_loader = self.datasets.get_train_loader()
        for b, batch_data in enumerate(data_loader):
            losses = self.train_on_batch(b, batch_data)
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

    def get_losses(self, batch_data):
        inputs, labels = batch_data
        outputs = self(inputs)
        losses={}
        for r in self.loss_funcs:
            losses[r] = self.loss_funcs[r](outputs, labels)
        return losses

    def get_dataset_losses(self, dataset):
        with torch.no_grad():
            self.train_mode(False)
            return self.get_losses(tuple(dataset.tensors))

    def set_datasets(self, datasets):
        self.datasets = datasets
        self.num_data = len(datasets.train_dataset)

    def initialize_model(self):
        t.initialize_model(self)

    def evaluate(self):
        losses = self.get_dataset_losses(self.datasets.train_dataset)
        if hasattr(self.datasets, 'test_dataset'):
            test_losses = self.get_dataset_losses(self.datasets.test_dataset)
            for r in test_losses:
                if 'test_'+r in self.reporters:
                    losses['test_'+r] = test_losses[r]
        if self.validation:
            val_losses = self.get_dataset_losses(self.datasets.val_dataset)
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
            else:
                if (losses['val_'+self.monitor] < self.monitor_loss) and not torch.isnan(losses['val_'+self.monitor]):
                    self.monitor_loss = losses['val_'+self.monitor]
                    self.monitor_losses = losses
                    torch.save(self.state_dict(), self.weights_file)

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

    def save_evaluate(self):
        if self.validation:
            t.save_evaluate(self.args.output, self.name, 'data_ver,data_name,exp_no,epochs,batch_size,%s,%s,fit_time\n' % (self.monitor, self.test_monitor),\
                [self.args.data_ver, self.args.data_name, self.args.exp_no, self.epochs,self.batch_size,\
                self.monitor_losses[self.monitor].item(), self.monitor_losses[self.test_monitor].item() if self.test_monitor in self.monitor_losses else 'None', self.fit_time])
        else:
            t.save_evaluate(self.args.output, self.name, 'data_ver,data_name,exp_no,epochs,batch_size,loss,fit_time\n',\
                [self.args.data_ver, self.args.data_name, self.args.exp_no, self.epochs,self.batch_size,\
                self.history['loss'][-1], self.fit_time])

acts = {
    'relu':torch.relu,
    'tanh':torch.tanh,
    'sigmoid':torch.sigmoid,
    'leakyrelu':torch.nn.functional.leaky_relu,
}

act_modules = {
    'relu':nn.ReLU(),
    'tanh':nn.Tanh(),
    'sigmoid':nn.Sigmoid(),
    'leakyrelu':torch.nn.modules.LeakyReLU(),
}

poolings={
    'max':nn.MaxPool2d(2),
    'avg':nn.AvgPool2d(2),
}