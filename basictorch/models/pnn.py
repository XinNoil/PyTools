from .base import *
from .dnn import DNN
from .losses import euclidean_error
import mtools as mt

distances={
    'sub':torch.sub,
    'euc':euclidean_error,
}

default_model_params={
    'distance':'sub', 
    'var_dim_1':False,
}

class PNN(Base): 
    def set_model_params(self, model_params):
        super().set_model_params(model_params, default_model_params)
        self.mean_dnn = DNN('mean_dnn', self.args, **model_params)
        if self.var_dim_1:
            del model_params['dim_y']
            self.var_dnn  = DNN('var_dnn',  self.args, dim_y=1, **model_params)
        else:
            self.var_dnn  = DNN('var_dnn',  self.args, **model_params) # , out_activation='relu'
        self.loss_funcs['loss'] = loss_funcs['nll']
        self.distance = distances[self.distance]
        self.mean_optimizer = optim.Adadelta(self.mean_dnn.parameters(), rho=0.95, eps=1e-7)
        self.var_optimizer  = optim.Adadelta(self.var_dnn.parameters(),  rho=0.95, eps=1e-7)
        self.optimizer = optim.Adadelta(self.parameters(), rho=0.95, eps=1e-7)

    def forward(self, x):
        return self.mean_dnn(x), self.var_dnn(x)
    
    def get_losses(self, batch_data):
        inputs, labels = batch_data
        outputs = self(inputs)
        nll = loss_funcs['nll'](outputs, labels, distance=self.distance)
        mee = loss_funcs['mee'](outputs[0], labels)
        return {'loss':nll, 'mee':mee}
    
    def train(self, batch_size=16, epochs=0, validation=True, reporters=['loss','mee','test_loss','test_mee'], monitor='mee', test_monitor=None, initialize=True):
        super().train(batch_size=batch_size, epochs=epochs, validation=validation, reporters=reporters, monitor=monitor, test_monitor=test_monitor, initialize=initialize)
    
    def save_end(self):
        super().save_end()
        x_test, y_test = self.datasets.test_dataset.tensors
        y_mean, y_var = self(x_test)
        err = euclidean_error(y_test, y_mean)
        output_tensor = torch.cat((y_test, y_mean, y_var, err.view(-1, 1)), dim=-1)
        mt.csvwrite(t.get_filename(self.args, 'test_result','csv'), output_tensor.detach().cpu().numpy())

models={
    'dnn':DNN,
    'pnn':PNN,
}

class Ensemble(Base):
    def __init__(self, name, args, ensemble_size=5, model='pnn', **model_params):
        super().__init__(name, args, ensemble_size=ensemble_size, model=model, **model_params)
    
    def set_model_params(self, model_params):
        super().set_model_params(model_params, default_model_params)
        self.model = models[self.model]
        self.loss_funcs['loss'] = loss_funcs['nll'] if issubclass(self.model, PNN) else loss_funcs['mee']
        self.distance = distances[self.distance]
        self.models = nn.ModuleList([self.model('model%d'%i, self.args, **model_params) for i in range(self.ensemble_size)])
    
    def forward(self, x):
        outputs = [model(x) for model in self.models]
        if issubclass(self.model, PNN):
            mean   = [output[0] for output in outputs]
            var    = [output[1] for output in outputs]
            mean_e = torch.mean(torch.stack(mean), 0)
            distances = torch.stack([self.distance(mean_, mean_e) for mean_ in mean])
            var_e  =  torch.var(distances, 0) + torch.mean(torch.stack(var),  0)
            return mean_e, var_e
        else:
            mean = torch.mean(torch.stack(outputs), 0)
            return mean
    
    def train_on_batch(self, b, batch_data):
        for model in self.models:
            model.train_on_batch(b, batch_data)
        return self.get_losses(batch_data)
    
    def get_losses(self, batch_data):
        inputs, labels = batch_data
        outputs = self(inputs)
        if issubclass(self.model, PNN):
            nll = loss_funcs['nll'](outputs, labels, distance=self.distance)
            mee = loss_funcs['mee'](outputs[0], labels)
            return {'loss':nll, 'mee':mee}
        else:
            mee = loss_funcs['mee'](outputs, labels)
            return {'loss':mee}
    
    def train(self, batch_size=16, epochs=0, validation=True, test_monitor=None, initialize=True):
        if issubclass(self.model, PNN):
            reporters=['loss','mee','test_loss','test_mee']
            monitor='mee'
        else:
            reporters=['loss','test_loss']
            monitor='loss'
        super().train(batch_size=batch_size, epochs=epochs, validation=validation, reporters=reporters, monitor=monitor, test_monitor=test_monitor, initialize=initialize)
    
    def save_end(self):
        super().save_end()
        if issubclass(self.model, PNN):
            x_test, y_test = self.datasets.test_dataset.tensors
            outputs = [model(x_test) for model in self.models]
            mean   = [output[0] for output in outputs]
            var    = [output[1] for output in outputs]
            y_mean = torch.mean(torch.stack(mean), 0)
            distances = torch.stack([self.distance(mean_, y_mean) for mean_ in mean])
            y_var  =  torch.var(distances, 0) + torch.mean(torch.stack(var),  0)
            err = euclidean_error(y_test, y_mean)
            output_tensor = torch.cat((y_test, y_mean, y_var, err.view(-1, 1), *mean, *var), dim=-1)
            mt.csvwrite(t.get_filename(self.args, 'test_result','csv'), output_tensor.detach().cpu().numpy())