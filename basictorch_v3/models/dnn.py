from torch import nn
from ..layers import act_modules, get_layers

class DNN(nn.Module):
    def __init__(self, name, dim_x, dim_y, layer_units, activations='relu', out_activation=None, dropouts=[0.0, 0.0], Layer = nn.Linear, **kwargs):
        super().__init__()
        self.name = name
        self.dropouts = dropouts
        if dropouts[0]>0:
            self.dropout_i = nn.Dropout(dropouts[0])
        self.layers = get_layers(dim_x, layer_units, Layer=Layer, **kwargs)
        self.activations = act_modules[activations]
        if dropouts[1]>0:
            self.dropout_h = nn.Dropout(dropouts[1])
        self.out_layer = Layer(layer_units[-1] if len(layer_units)>0 else dim_x, dim_y, **kwargs) if dim_y else nn.Identity()
        if out_activation is not None:
            self.out_activation = act_modules[out_activation]
    
    def forward(self, x, *args, **kwargs):
        if self.dropouts[0]>0:
            x = self.dropout_i(x)
        if self.dropouts[1]>0:
            for layer in self.layers:
                x = layer(x, *args, **kwargs)
                x = self.activations(x)
                x = self.dropout_h(x)
        else:
            for layer in self.layers:
                x = layer(x, *args, **kwargs)
                x = self.activations(x)
        x = self.out_layer(x)
        if hasattr(self, 'out_activation'):
            x = self.out_activation(x)
        return x
