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
            self.activations = [model_params['activations'] for x in model_params['layer_units']]
        self.out_activation = model_params['out_activation']
        self.optimizer = optim.Adadelta(self.parameters(), rho=0.95, eps=1e-7)

    def build_model(self):
        if self.spectral_normalization:
            self.layers = t.get_layers(self.dim_x, self.layer_units, SNLinear)
            self.out_layer = SNLinear(self.layer_units[-1], self.dim_y)
        else:
            self.layers = t.get_layers(self.dim_x, self.layer_units)
            self.out_layer = nn.Linear(self.layer_units[-1], self.dim_y)
        if self.dropouts[0]>0:
            self.dropout_i = nn.Dropout(self.dropouts[0])
        if self.dropouts[1]>0:
            self.dropout_h = nn.Dropout(self.dropouts[1])

    def forward(self, x):
        x = self.dropout_i(x) if self.dropouts[0]>0 else x
        if hasattr(self, 'scale_h1'):
            x = self.dropout_h(acts[self.activations[0]](self.layers[0](x))*self.scale_h1) if self.dropouts[1]>0 else acts[self.activations[0]](self.layers[0](x))*self.scale_h1
            for layer, activation in zip(self.layers[1:], self.activations[1:]):
                x = self.dropout_h(acts[activation](layer(x))) if self.dropouts[1]>0 else acts[activation](layer(x))
        else:
            for layer, activation in zip(self.layers, self.activations):
                x = self.dropout_h(acts[activation](layer(x))) if self.dropouts[1]>0 else acts[activation](layer(x))
        x = acts[self.out_activation](self.out_layer(x)) if self.out_activation else self.out_layer(x)
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