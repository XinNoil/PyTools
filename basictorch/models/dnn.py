from torch.utils.data.dataset import Dataset
from .base import *

default_model_params={
    'dim_x':None, 
    'dim_y':None, 
    'layer_units':[256,128,32], 
    'activations':'relu', 
    'out_activation':None, 
    'dropouts':[0.0, 0.0], 
    'loss_func':'mee', 
    'monitor':'mee',
    'spectral':False,
}

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
    # priority: model_params > args_params > default_model_params
    def set_args_params(self):
        self._set_args_params([
            'layer_units','activations','out_activation','dropouts','loss_func','monitor'
            ])
        super().set_args_params()
        
    def set_model_params(self, model_params, _default_model_params={}):
        super().set_model_params(model_params, t.get_model_params(_default_model_params, default_model_params), is_build_model=True)

    def build_model(self):
        self.layer_units.insert(0, self.dim_x)
        self.loss_funcs['loss'] = loss_funcs[self.loss_func]
        if self.monitor != 'loss':
            self.loss_funcs[self.monitor] = loss_funcs[self.monitor]
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
        self.optimizer = optim.Adadelta(self.parameters(), rho=0.95, eps=1e-7)

def train_dnn(args, Ds, dnn=None, model_params={}, train_params={}, func=None, func_params={}):
    if not dnn:
        dnn = DNN
    for e in range(args.trails):
        if hasattr(Ds, 're_split'):
            Ds.re_split()
        args.exp_no = t.get_exp_no(args, e+1)
        model = dnn('dnn', args,
                dim_x=Ds.train_dataset.tensors[0].shape[1], 
                dim_y=Ds.train_dataset.tensors[1].shape[1],
                layer_units = args.layer_units,
                dropouts = args.dropouts,
                **model_params
            ).to(t.device)
        print(model)
        model.set_datasets(Ds)
        if hasattr(args, 'load_model'):
            if args.load_model:
                model.load_model()
                model.apply_func(func, func_params)
                continue
        model.train(batch_size=args.batch_size, epochs=args.epochs, **train_params)
        model.apply_func(func, func_params)
    return model