from .base import *
from .dnn import DNN
from .layers import SNLinear

class DNN_scale(Base):
    def __init__(self, **model_params):
        super(DNN_scale, self).__init__(*model_params)

    def forward(self, x):
        x = self.dropout_i(x) if self.dropouts[0]>0 else x
        x = self.dropout_h(acts[self.activations[0]](self.layers[0](x))*self.scale_h1) if self.dropouts[1]>0 else acts[self.activations[0]](self.layers[0](x))*self.scale_h1
        for layer, activation in zip(self.layers[1:], self.activations[1:]):
            x = self.dropout_h(acts[activation](layer(x))) if self.dropouts[1]>0 else acts[activation](layer(x))
        x = acts[self.out_activation](self.out_layer(x)) if self.out_activation else self.out_layer(x)
        return x
