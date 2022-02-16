from torch import nn
from ..layers import act_modules, poolings, View, drops, norm_layers
from .dnn import DNN

def get_block(con_in, con_out, activations, pooling, norm_layer=None, kernel_size=5, stride=1, padding=2, pool_size=2, drop_type='dropout', dropout=0):
    return nn.Sequential(
        nn.Conv2d(con_in, con_out, kernel_size, stride, padding),
        norm_layers[norm_layer](con_out),
        act_modules[activations],
        poolings[pooling](pool_size),
        drops[drop_type](dropout) if dropout else nn.Identity()
    )

def get_d_block(con_in, con_out, activations, norm_layer=None, kernel_size=5, stride=1, padding=2, upsample_mode='nearest'):
    return nn.Sequential(
        nn.ConvTranspose2d(con_in, con_out, kernel_size, stride, padding),
        norm_layers[norm_layer](con_out) if norm_layer else nn.Identity(),
        act_modules[activations],
        nn.Upsample(scale_factor=2, mode=upsample_mode)
    )

class CNN(nn.Module):
    def __init__(self, name, dim_x, dim_y, layer_units, activations='relu', out_activation=None, dropouts=[0.0, 0.0], cons=[1, 8, 16], dim=32, pooling='max', norm_layer=None, avgpool=False, drop_type='dropout', flatten=None, mlp=None, kernel_size=5, stride=1, padding=2, pool_size=2, conv_dropout=0):
        super().__init__()
        self.name  = name
        self.dropout_i = drops[drop_type](dropouts[0]) if dropouts[0]>0 else nn.Identity()
        self.reshape = nn.Linear(dim_x, cons[0]*dim*dim) if dim_x else nn.Identity()

        self.view = View(-1, cons[0], dim, dim)
        self.blocks = nn.Sequential()
        for i, con_in, con_out in zip(range(len(cons)-1), cons[:-1], cons[1:]):
            self.blocks.add_module('conv%d'%i, get_block(con_in, con_out, activations, pooling, norm_layer, kernel_size, stride, padding, pool_size, 'dropout', conv_dropout))
        self.avgpool = nn.AdaptiveAvgPool2d((1, 1)) if avgpool else nn.Identity()
        self.flatten = flatten if flatten is not None else nn.modules.Flatten()
        mlp_dim_x = cons[-1] * (1 if avgpool else (int(dim/(2**(len(cons)-1))))**2)
        self.mlp = mlp if mlp is not None else DNN('mlp', mlp_dim_x, dim_y, layer_units, activations, out_activation, [dropouts[1], dropouts[1]])
    
    def forward(self, x, *args, **kwargs):
        h = self.dropout_i(x)
        h = self.reshape(h)
        h = self.view(h)
        h = self.blocks(h)
        h = self.avgpool(h)
        h = self.flatten(h)
        h = self.mlp(h)
        return h

class DCNN(nn.Module):
    def __init__(self, name, dim_x, dim_y, layer_units, activations='relu', out_activation=None, dropouts=[0.0, 0.0], cons=[16, 8, 1], dim=32, norm_layer=None, flatten=None, mlp=None, kernel_size=5, stride=1, padding=2):
        super().__init__()
        self.name  = name
        dim_co = int(dim/(2**(len(cons)-1)))
        self.mlp = mlp if mlp is not None else DNN('mlp', dim_x, cons[0]*dim_co*dim_co, layer_units, activations, out_activation, dropouts)
        self.view =  View(-1, cons[0], dim_co, dim_co)
        self.blocks = nn.Sequential()
        for i, con_in, con_out in zip(range(len(cons)-1), cons[:-1], cons[1:]):
            self.blocks.add_module('conv%d'%i, get_d_block(con_in, con_out, activations, norm_layer, kernel_size, stride, padding))
        self.flatten = flatten if flatten is not None else nn.modules.Flatten()
        self.reshape = nn.Linear(cons[-1]*dim*dim, dim_y) if dim_y else nn.Identity()
    
    def forward(self, x, *args, **kwargs):
        h = self.mlp(x)
        h = self.view(h)
        h = self.blocks(h)
        h = self.flatten(h)
        h = self.reshape(h)
        return h
