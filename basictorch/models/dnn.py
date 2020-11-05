from .base import *
from .layers import SNLinear

class DNN(Base):
    def __init__(self, name, args, set_model_params=True, dim_x=None, dim_y=None, layer_units=[256,128,32], activations=None, out_activation=None, dropouts=[0.0, 0.0], loss_func='mee', spectral_normalization=False):
        super(DNN, self).__init__(name, args, dim_x=dim_x, dim_y=dim_y, layer_units=layer_units, activations=activations, out_activation=out_activation, dropouts=dropouts, loss_func=loss_func, spectral_normalization=spectral_normalization) if set_model_params else super(DNN, self).__init__(name, args)

    def set_model_params(self, model_params):
        self.model_params = model_params
        self.dim_x = model_params['dim_x']
        self.dim_y = model_params['dim_y']
        self.layer_units = model_params['layer_units']
        self.dropouts = model_params['dropouts']
        self.spectral_normalization = model_params['spectral_normalization']
        self.build_model()
        self.loss_funcs['loss'] = loss_funcs[model_params['loss_func']]
        if model_params['activations'] is None:
            self.activations = ['relu' for x in model_params['layer_units']]
        else:
            self.activations = model_params['activations']
        self.out_activation = model_params['out_activation']
        self.optimizer = optim.Adadelta(self.parameters(), rho=0.95, eps=1e-7)

    def build_model(self):
        if self.spectral_normalization:
            self.layers = t.get_layers(self.dim_x, self.layer_units, SNLinear)
            self.out_layer = SNLinear(self.layer_units[-1], self.dim_y)
        else:
            self.layers = t.get_layers(self.dim_x, self.layer_units)
            self.out_layer = nn.Linear(self.layer_units[-1], self.dim_y)

    def forward(self, x):
        for layer, activation in zip(self.layers, self.activations):
            x = acts[activation](layer(x))
        x = self.out_layer(x)
        if self.out_activation is not None:
            x = acts[self.out_activation](x)
        return x

def train_dnn(args, Ds):
    for e in range(args.trails):
        args.exp_no = t.get_exp_no(args, e+1)
        model = DNN('dnn', args,
                dim_x=Ds.train_dataset.tensors[0].shape[1], 
                dim_y=Ds.train_dataset.tensors[1].shape[1],
                layer_units = args.layer_units,
                dropouts = args.dropouts,
            ).to(t.device)
        print(model)
        model.set_datasets(Ds)
        model.train(batch_size=args.batch_size, epochs=args.epochs, initialize=True)
    return model