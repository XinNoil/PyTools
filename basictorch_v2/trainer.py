import os, sys, torch, time, inspect
import torch.optim as optim
import basictorch_v2.tools as t
from .losses import loss_funcs
from .tools import Base, OutputManger
from mtools import join_path, save_json, get_git_info, write_file, not_none, np

def get_optim(parameters, type, **kwargs):
    if type=='Adadelta':
        rho = kwargs['rho'] if 'rho' in kwargs else 0.95
        eps = kwargs['eps'] if 'eps' in kwargs else 1e-7
        return optim.Adadelta(parameters, rho=rho, eps=eps, **kwargs)
    elif type=='Adam':
        lr = kwargs['lr'] if 'lr' in kwargs else 2.0e-4
        betas = kwargs['betas'] if 'betas' in kwargs else (0.5, 0.999)
        return optim.Adam(parameters, lr=lr, betas=betas)

def get_sub_dict(kwargs, names):
    sub_kwargs = {}
    for name in names:
        if name in kwargs:
            sub_kwargs[name] = kwargs[name]
    return sub_kwargs

class Trainer(Base):
    # priority: kw_params > args_params > default_params
    # def __init__(self, name, args, outM=None, model=None, default_args={}, args_names=[], **kwargs):
    #     super().__init__(name, args, outM=outM, model=model, default_args={**{}, **default_args}, args_names=list(set(args_names+[])), **kwargs)

    def __init__(self, name, args, outM=None, model=None, default_args={}, args_names=[], **kwargs):
        self.outM = OutputManger(args, **get_sub_dict(kwargs, ['output', 'model_name', 'data_name', 'data_ver', 'data_postfix', 'feature_mode', 'e'])) if outM is None else outM
        self.model = model
        default_args = {**{'loss_func':loss_funcs['mee']}, **default_args}
        args_names = list(set(args_names + []))
        super().__init__(name, args, default_args=default_args, args_names=args_names, **kwargs)

    def fit(self, model, optimizer, datasets, batch_size, epochs, validation=True, monitor='loss', test_monitor='loss', test_reporters=[], initialize=True, batch_i=[0,1], batch_size_eval=128, **kwargs):
        self.model = model
        self.optimizer = optimizer
        self.datasets = datasets
        self.batch_size = batch_size
        self.epochs = epochs
        self.validation = validation
        self.monitor = monitor
        self.test_monitor = test_monitor
        self.test_reporters = test_reporters
        self.initialize = initialize
        self.batch_i = batch_i
        self.batch_size_eval = batch_size_eval
        t.set_params(self, kwargs)
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
    
    def on_train_begin(self):
        print(self.name)
        print('\n\n--- START TRAINING ---\n\n')
        t.set_weight_file(self)
        self.history = {}
        self.train_fit_time = time.process_time()
        self.b = 0
        self.epoch = -1
        self.initialize_model()
        self.losses = self.evaluate()
        self.check_validation()
        t.print_epoch(self.epoch, self.epochs, self.losses, 0)

    def on_epoch_begin(self, epoch):
        self.model.train(True)
        self.epoch_start_time = time.process_time()
        self.epoch = epoch
        self.data_loader = self.datasets.get_data_loader('train')
    
    def train_on_epoch(self):
        sum_losses = {}
        for b, batch_data in enumerate(self.data_loader):
            self.b = b            
            losses = t.detach_losses(self.train_on_batch(b, batch_data))
            self.print_batch(b, losses)
            sum_losses = t.add_losses(sum_losses, losses)
        self.losses = t.div_losses(sum_losses, len(self.data_loader))
    
    def train_on_batch(self, b, batch_data):
        self.optimizer.zero_grad()
        losses = self.get_losses(batch_data)
        losses['loss'].backward()
        self.optimizer.step()
        return losses

    def on_epoch_end(self):
        epoch_time = time.process_time() - self.epoch_start_time
        self.losses = self.evaluate(self.losses)
        self.check_validation()
        self.add_losses_to_history()
        t.print_epoch(self.epoch, self.epochs, self.losses, epoch_time)
    
    def on_train_end(self):
        self.fit_time = time.process_time() - self.train_fit_time
        self.model.load_state_dict(torch.load(self.weights_file))
        self.save_end()
        print('\n\n--- FINISH TRAINING ---\n\n')
    
    def initialize_model(self):
        if self.initialize:
            t.reset_parameters(self.model)
            t.initialize_model(self.model)
    
    def print_batch(self, b, losses):
        if hasattr(self.args, 'print_batch') and not self.args.print_batch:
            sys.stdout.flush()
        else:
            t.print_batch(self.epoch, self.epochs, b, self.batch_size, len(self.datasets.train_dataset), losses)

    def get_losses(self, batch_data):
        inputs, labels = t.unpack_batch(batch_data, self.batch_i)
        outputs = self.model(inputs)
        return self._get_losses(labels, outputs)
    
    def _get_losses(self, labels, outputs):
        return {'loss':self.loss_func(labels, outputs)}

    def get_dataset_losses(self, data_loader):
        losses = {}
        for b,batch_data in enumerate(data_loader):
            _losses = self.get_losses(batch_data)
            losses = t.add_losses(losses, _losses)
        last_batch_num = np.max([b.shape[0] for b in batch_data])
        if b>0:
            p = 1.0-(last_batch_num/self.batch_size_eval)
            for loss in _losses:
                _losses[loss] *= -p
            losses = t.add_losses(losses, _losses)
            div_num = len(data_loader) - p
        else:
            div_num = len(data_loader)
        return t.div_losses(losses, div_num)

    def set_datasets(self, datasets):
        self.datasets = datasets

    def evaluate(self, losses=None, batch_size=None):
        batch_size = batch_size if batch_size else self.batch_size_eval
        with torch.no_grad():
            self.model.train(False)
            if (losses is None) or (not self.validation):
                losses = self.get_dataset_losses(self.datasets.get_data_loader('train', batch_size=batch_size))
            if hasattr(self.datasets, 'test_dataset'):
                test_losses = self.get_dataset_losses(self.datasets.get_data_loader('test', batch_size=batch_size))                
                losses['test_'+self.test_monitor] = test_losses[self.test_monitor]
                for r in test_losses:
                    if r in self.test_reporters:
                        losses['test_'+r] = test_losses[r]
            if self.validation:
                self.val_losses = self.get_dataset_losses(self.datasets.get_data_loader('val', batch_size=batch_size))
                losses['val_'+self.monitor] = self.val_losses[self.monitor]
            return losses

    def check_validation(self):
        monitor = 'val_'+self.monitor if self.validation else self.monitor
        if (self.epoch  == -1) or ((self.losses[monitor] < self.monitor_loss) and not torch.isnan(self.losses[monitor])):
            self.save_checkpoint()
        
    def save_checkpoint(self):
        monitor = 'val_'+self.monitor if self.validation else self.monitor
        self.monitor_loss = self.losses[monitor]
        self.monitor_losses = self.losses
        self.val_epoch = self.epoch
        torch.save(self.model.state_dict(), self.weights_file)

    def add_losses_to_history(self):
        for r in list(self.losses.keys()):
            if r in self.history:
                self.history[r].append(self.losses[r].item())
            else:
                self.history[r] = [self.losses[r].item()]

    def save_end(self):
        self.save_model()
        t.save_args(self.outM, self.args)
        t.curve_plot(self.outM, 'curve_%s' % self.name, self.history)
        if os.path.exists(join_path('configs','git.json')):
            save_json(self.outM.get_filename('gitinfo', 'json'), get_git_info(join_path('configs','git.json')))
        if hasattr(self, 'save_evaluate_func'):
            self.save_evaluate_func(self)
        else:
            self.save_evaluate()

    def save_model(self, model=None, outM=None, postfix='', filename=None, model_name=None):
        model = not_none(model, self.model)
        write_file(self.outM.get_filename('model_arch_%s%s'%(model.name, postfix), 'txt', by_exp_no=False), str(model).split('\n'))
        write_file(self.outM.get_filename('model_class_%s%s'%(model.name, postfix), 'txt', by_exp_no=False), str(inspect.getsource(type(model))).split('\n'))
        t.save_model(model, not_none(outM, self.outM), postfix=postfix, filename=filename, model_name=model_name)
    
    def load_model(self, model=None, outM=None, postfix='', filename=None, model_name=None):
        t.load_model(not_none(model, self.model), not_none(outM, self.outM), postfix=postfix, filename=filename, model_name=model_name)

    def save_evaluate(self, postfix='', head=None, varList=None):
        if head is None:
            if self.validation:
                head = 'data_ver,data_name,exp_no,epochs,batch_size,%s,%s,fit_time,val_epoch\n' % (self.monitor, 'test_%s'%self.test_monitor)
                varList = [self.outM.data_ver, self.outM.data_name, self.outM.exp_no, self.epochs, self.batch_size,\
                    self.monitor_losses[self.monitor].item(), self.monitor_losses['test_%s'%self.test_monitor].item() if 'test_%s'%self.test_monitor in self.monitor_losses else 'None', self.fit_time, self.val_epoch]
            else:
                head = 'data_ver,data_name,exp_no,epochs,batch_size,loss,fit_time\n'
                varList = [self.outM.data_ver, self.outM.data_name, self.outM.exp_no, self.epochs, self.batch_size, self.history['loss'][-1], self.fit_time]
        if hasattr(self.args, 'run_i'):
            head = 'run_i,'+head
            varList.insert(0, self.args.run_i)
        t.save_evaluate(self.outM.output, '%s%s'%(self.name, postfix), head, varList)
