from torch import nn
from ..tools import get_layers
from ..layers import act_modules, drops

class DNN(nn.Module):
    def __init__(self, name, dim_x, dim_y, layer_units, activations='relu', out_activation=None, dropouts=[0.0, 0.0], drop_type='dropout', Layer = nn.Linear, pre_layer=nn.Identity(), **kwargs):
        super().__init__()
        self.name = name
        self.dropout_i = drops[drop_type](dropouts[0]) if dropouts[0]>0 else nn.Identity()
        self.pre_layer = pre_layer
        self.layers = get_layers(dim_x, layer_units, Layer=Layer, **kwargs)
        self.activations = act_modules[activations]
        self.dropout_h = nn.Dropout(dropouts[1]) if dropouts[1]>0 else nn.Identity()
        self.out_layer = Layer(layer_units[-1] if len(layer_units)>0 else dim_x, dim_y, **kwargs) if dim_y else nn.Identity()
        self.out_activation = act_modules[out_activation] if out_activation is not None else nn.Identity()
    
    def forward(self, x, *args, **kwargs):
        h = self.dropout_i(x)
        h = self.pre_layer(h)
        for layer in self.layers:
            h = layer(h)
            h = self.activations(h)
            h = self.dropout_h(h)
        h = self.out_layer(h)
        h = self.out_activation(h)
        return h
