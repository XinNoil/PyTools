from .base import *
from .dnn import DNN
from .losses import euclidean_error
import mtools as mt

default_model_params={
    'dim_x':None, 
    'dim_y':None, 
    'layer_units':[256,128,32], 
    'ensemble_size':10,
    'activations':'relu', 
    'out_activation':None, 
    'dropouts':[0.1, 0], 
    'loss_func':'mee', 
    'spectral':False,
}

class MCPNN(DNN):
    def set_model_params(self, model_params):
        Base.set_model_params(self, model_params, default_model_params)
        self.distance = torch.sub
        self.layer_units.insert(0, self.dim_x)
        self.loss_funcs['loss'] = loss_funcs[self.loss_func]
        self.build_model()
        self.optimizer = optim.Adadelta(self.parameters(), rho=0.95, eps=1e-7)

    def predict(self, x):
        mean = [self(x) for i in range(self.ensemble_size)]
        mean_e = torch.mean(torch.stack(mean), 0)
        distances = torch.stack([self.distance(mean_, mean_e) for mean_ in mean])
        var_e  =  torch.var(distances, 0)
        return mean_e, var_e

    def get_losses(self, batch_data):
        self.train_mode(True)
        inputs, labels = batch_data
        outputs = self.predict(inputs)
        output = self(inputs)
        loss = self.loss_funcs['loss'](output, labels)
        loss_mean = self.loss_funcs['loss'](outputs[0], labels)
        return {'loss':loss, 'loss_mean':loss_mean}
    
    def save_end(self):
        super().save_end()
        x_test, y_test = self.datasets.test_dataset.tensors
        self.train_mode(True)
        y_mean, y_var = self.predict(x_test)
        err = euclidean_error(y_test, y_mean)
        output_tensor = torch.cat((y_test, y_mean, y_var, err.view(-1, 1)), dim=-1)
        mt.csvwrite(t.get_filename(self.args, 'test_result','csv'), output_tensor.detach().cpu().numpy())
        