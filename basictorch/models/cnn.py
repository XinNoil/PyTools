from .base import *
from .layers import SNLinear,SNConv2d

class CNN(Base):
    def __init__(self, name, args, set_model_params=True, con1=8, con2=16, dim_conv_i=32, dim_x=None, dim_y=None, layer_units=[], activations=None, out_activation=None, dropouts=[0.0, 0.0], loss_func='mee', spectral=False):
        super(CNN, self).__init__(name, args, con1=con1, con2=con2, dim_conv_i=dim_conv_i, dim_x=dim_x, dim_y=dim_y, layer_units=layer_units, activations=activations, out_activation=out_activation, dropouts=dropouts, loss_func=loss_func, spectral=spectral) if set_model_params else super(CNN, self).__init__(name, args)
    
    def set_model_params(self, model_params):
        self.dim = model_params['dim_conv_i']
        self.con1 = model_params['con1']
        self.con2 = model_params['con2']
        self.dim_co = int(self.dim/4)
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
            Linear = SNLinear
            Conv2d = SNConv2d
        else:
            Linear = nn.Linear
            Conv2d = nn.Conv2d
        self.reshape = Linear(self.dim_x, self.dim*self.dim)
        self.conv1 = nn.Sequential(  # input shape (1, 32, 32)
            Conv2d(1, self.con1, 5, 1, 2),      # output shape (con1, 32, 32)
            nn.ReLU(),    # activation
            nn.MaxPool2d(2),    # output shape (con1, 16, 16)
        )
        self.conv2 = nn.Sequential(  # input shape (con1, 16, 16)
            Conv2d(self.con1, self.con2, 5, 1, 2),  # output shape (con2, 16, 16)
            nn.ReLU(),  # activation
            nn.MaxPool2d(2),  # output shape (con2, 8, 8)
        )
        if len(self.layer_units)>0:
            self.layers = t.get_layers(self.con2 * self.dim_co * self.dim_co, self.layer_units, Linear)
            self.out_layer = Linear(self.layer_units[-1], self.dim_y)
        else:
            self.out_layer = Linear(self.con2 * self.dim_co * self.dim_co, self.dim_y)

    def forward(self, x):
        x = self.reshape(x)
        x = x.view(-1, 1, self.dim, self.dim)
        x = self.conv1(x)
        x = self.conv2(x)
        x = x.view(-1, self.con2 * self.dim_co * self.dim_co)
        for layer, activation in zip(self.layers, self.activations):
            x = acts[activation](layer(x))
        x = self.out_layer(x)
        if self.out_activation is not None:
            x = acts[self.out_activation](x)
        return x
