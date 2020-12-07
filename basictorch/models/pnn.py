from .base import *
from .dnn import DNN
from .losses import euclidean_error

default_model_params={
    'distance':torch.sub, 
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
            self.var_dnn  = DNN('var_dnn',  self.args, **model_params)
        self.loss_funcs['loss'] = loss_funcs['nll']
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
    
    # def train_on_batch(self, b, batch_data):
    #     inputs, labels = batch_data
    #     self.mean_optimizer.zero_grad()
    #     mean = self.mean_dnn(inputs)
    #     with torch.no_grad():
    #         var = self.var_dnn(inputs)
    #     loss = self.loss_funcs['loss']((mean, var), labels)
    #     loss.backward()
    #     self.mean_optimizer.step()

    #     self.var_optimizer.zero_grad()
    #     var = self.var_dnn(inputs)
    #     with torch.no_grad():
    #         mean = self.mean_dnn(inputs)
    #     loss = self.loss_funcs['loss']((mean, var), labels)
    #     loss.backward()
    #     self.var_optimizer.step()

    #     return self.get_losses(batch_data)

class Ensemble(Base):
    def __init__(self, name, args, ensemble_size=5, model=DNN, distance=euclidean_error, var_dim_1=True, **model_params):
        super().__init__(name, args, ensemble_size=ensemble_size, model=model, distance=distance, var_dim_1=var_dim_1, **model_params)
    
    def set_model_params(self, model_params):
        self.model_params = model_params
        ensemble_size = model_params['ensemble_size']
        self.model = model_params['model']
        self.distance = model_params['distance']
        self.models = nn.ModuleList([self.model('model%d'%i, self.args, **model_params) for i in range(ensemble_size)])
    
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
    
    def save_model(self):
        pass
