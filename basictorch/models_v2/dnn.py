from torch.utils.data.dataset import Dataset
from .base import *

class dnn(nn.Module):
    def __init__(self, dim_x, dim_y, layer_units):
        super().__init__()
        self.layers = t.get_layers(dim_x, layer_units)
        self.out_layer = nn.Linear(layer_units[-1], dim_y)
    
    def forward(self, x):
        for layer in self.layers:
            x = torch.relu(layer(x))
        return self.out_layer(x)

class DNN(Base):
    def set_params(self, model_params):
        self.add_default_params({
            'dim_x':None, 
            'dim_y':None, 
            'layer_units':[256,128,32], 
            'activations':'relu', 
            'out_activation':None, 
            'dropouts':[0.0, 0.0], 
            'loss_func':'mee', 
            'monitor':'mee',
            'spectral':False,
        })
        self.add_args_params(['layer_units','activations','out_activation','dropouts','loss_func','monitor'])
        super().set_params(model_params)

    def build_model(self, is_set_optim=True):
        self.layer_units.insert(0, self.dim_x)
        self.set_loss_funcs()
        if self.monitor != 'loss':
            self.loss_funcs[self.monitor] = get_loss_func(self.loss_func, self.args)
        self.sequential = nn.Sequential()
        if self.dropouts[0]>0:
            self.sequential.add_module('dropout_i', nn.Dropout(self.dropouts[0]))
        for l in range(len(self.layer_units)-1):
            self.sequential.add_module('layer%d'%l, nn.Linear(self.layer_units[l], self.layer_units[l+1]))
            self.sequential.add_module('%s%d'%(self.activations, l), act_modules[self.activations])
            if self.dropouts[1]>0:
                self.sequential.add_module('dropout_%d'%l, nn.Dropout(self.dropouts[1]))
        if self.dim_y:
            self.sequential.add_module('out_layer', nn.Linear(self.layer_units[-1], self.dim_y))
            if self.out_activation:
                self.sequential.add_module(self.out_activation, act_modules[self.out_activation])
        else:
            self.dim_y = self.layer_units[-1]
        if self.spectral:
            self.apply(t.spectral_norm)
        if is_set_optim:
            self.optimizer = optim.Adadelta(self.parameters(), rho=0.95, eps=1e-7)
    
    def get_intra_feature(self, x, name):
        for _name, _module in self.sequential._modules.items():
            x = _module(x)
            if _name == name:
                break
