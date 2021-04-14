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
        self.args = args # args.output, args.data_name, args.data_ver, args.exp_no
        self.loss_funcs = {}
        self.args_params = []
        self.set_args_params()
        if set_model_params:
            self.set_model_params(model_params)
    
    def train_mode(self, mode):
        super().train(mode)
    
    def set_args_params(self):
        self._set_args_params([])
        # super().set_args_params()
        
    def _set_args_params(self, args_params):
        for param in args_params:
            if param not in self.args_params:
                self.args_params.append(param)

    def set_model_params(self, model_params, default_model_params={}):
        for param in self.args_params:
            if hasattr(self.args, param):
                default_model_params[param] = self.args.__dict__[param].copy() if isinstance(self.args.__dict__[param], list) else self.args.__dict__[param]
        self.model_params = self.get_model_params(model_params, default_model_params)
        for param in model_params:
            self.__dict__[param] = model_params[param].copy() if isinstance(model_params[param], list) else model_params[param]
        if len(self.args_params):
            print('%s args_params: %s' % (self.name, str(self.args_params)))
        print('%s model_params: %s' % (self.name, str(self.model_params)))
    
    def get_model_params(self, model_params, default_model_params):
        for param in default_model_params:
            if param not in model_params:
                model_params[param] = default_model_params[param]
        return model_params
            
    def build_model(self):
        self.sequential = nn.Sequential()

    def forward(self, x):
        return self.sequential(x)

    def train(self, batch_size=0, epochs=0, validation=True, reporters=['loss','test_loss'], monitor='loss', test_monitor=None, initialize=True, max_sub_size=1e3):
        if type(batch_size) != bool:
            if epochs>0:
                self.batch_size = batch_size
                self.epochs = epochs
                self.validation = validation
                self.reporters = reporters
                self.monitor = monitor
                self.test_monitor = test_monitor if test_monitor else 'test_%s' % self.monitor
                self.initialize = initialize
                self.max_sub_size = max_sub_size
                self.on_train_begin()
                for e in range(self.epochs):
                    self.on_epoch_begin(e)
                    self.train_on_epoch()
                    self.on_epoch_end()
                self.on_train_end()
            else:
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

    def get_losses(self, batch_data):
        inputs, labels = batch_data
        outputs = self(inputs)
        losses={}
        for r in self.loss_funcs:
            losses[r] = self.loss_funcs[r](outputs, labels)
        return losses

    def get_dataset_losses(self, dataset):
        batch_data=tuple(dataset.tensors)
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
            test_losses = self.get_dataset_losses(self.datasets.test_dataset)
            for r in test_losses:
                if 'test_'+r in self.reporters: #如果要报道的话，就把他加入losses里面
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
                self.val_epoch = self.epoch
            else:
                if (losses['val_'+self.monitor] < self.monitor_loss) and not torch.isnan(losses['val_'+self.monitor]):
                    self.monitor_loss = losses['val_'+self.monitor]
                    self.monitor_losses = losses
                    torch.save(self.state_dict(), self.weights_file)
                    self.val_epoch = self.epoch

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
            head = 'data_ver,data_name,exp_no,epochs,batch_size,%s,%s,fit_time,val_epoch\n' % (self.monitor, self.test_monitor)
            varList = [self.args.data_ver, self.args.data_name, self.args.exp_no, self.epochs,self.batch_size,\
                self.monitor_losses[self.monitor].item(), self.monitor_losses[self.test_monitor].item() if self.test_monitor in self.monitor_losses else 'None', self.fit_time, self.val_epoch]
        else:
            head = 'data_ver,data_name,exp_no,epochs,batch_size,loss,fit_time\n'
            varList = [self.args.data_ver, self.args.data_name, self.args.exp_no, self.epochs,self.batch_size,\
                self.history['loss'][-1], self.fit_time]
        if hasattr(self.args, 'run_i'):
            head = 'run_i,'+head
            varList.insert(0, self.args.run_i)
        t.save_evaluate(self.args.output, self.name, head, varList)

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
    
    def get_dataset_losses(self, dataset):
        batch_data = tuple(dataset.tensors)+tuple(self.datasets.get_unlab_dataset(dataset).tensors)
        return self.get_batchdata_losses(batch_data)
    
    def get_losses(self, batch_data):
        inputs, labels, unlabs = batch_data
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