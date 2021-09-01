from .base import *
from .layers import View

norm_layers = {
    'batchnorm':nn.BatchNorm2d,
}

def get_block(con_in, con_out, activations, pooling, norm_layer=None):
    if norm_layer:
        return nn.Sequential(
            nn.Conv2d(con_in, con_out, 5, 1, 2),
            norm_layers[norm_layer](con_out),
            act_modules[activations],
            pooling,
        )
    else:
        return nn.Sequential(
            nn.Conv2d(con_in, con_out, 5, 1, 2),
            act_modules[activations],
            pooling,
        )

class CNN(Base):
    def set_params(self, model_params):
        self.add_default_params({
            'cons':[1,8,16],
            'dim':32,
            'dim_x':None, 
            'dim_y':None, 
            'layer_units':[], 
            'activations':'relu', 
            'out_activation':None, 
            'dropouts':[0.0, 0.0], 
            'loss_func':'mee', 
            'spectral':False,
            'pooling':'max',
            'norm_layer':None,
            'avgpool':False,
        })
        super().set_params(model_params)

    def build_model(self):
        self.dim_co = int(self.dim/(2**(len(self.cons)-1)))
        self.layer_units.insert(0, self.cons[-1] if self.avgpool else self.cons[-1] * self.dim_co * self.dim_co)
        self.pooling = poolings[self.pooling]
        self.loss_funcs['loss'] = loss_funcs[self.loss_func]

        self.sequential = nn.Sequential()
        if self.dropouts[0] > 0:
            self.sequential.add_module('dropout_i', nn.Dropout(self.dropouts[0]))
        if self.dim_x:
            self.sequential.add_module('reshape', nn.Linear(self.dim_x, self.cons[0]*self.dim*self.dim))
        self.sequential.add_module('view', View(-1, self.cons[0], self.dim, self.dim))
        for i, con_in, con_out in zip(range(len(self.cons)-1), self.cons[:-1], self.cons[1:]):
            self.sequential.add_module('conv%d'%i, get_block(con_in, con_out, self.activations, self.pooling, self.norm_layer))
        if self.avgpool:
            self.sequential.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
        self.sequential.add_module('flatten', torch.nn.modules.Flatten())
        for i in range(len(self.layer_units)-1):
            self.sequential.add_module('layer%d'%(i), nn.Linear(self.layer_units[i], self.layer_units[i+1]))
            self.sequential.add_module('%s%d'%(self.activations, i), act_modules[self.activations])
        if self.dim_y:
            if self.avgpool:
                self.sequential.add_module('out_layer', nn.Linear(self.cons[-1], self.dim_y))
            else:
                self.sequential.add_module('out_layer', nn.Linear(self.layer_units[-1], self.dim_y))
        else:
            self.dim_y = self.cons[-1] if self.avgpool else int(self.cons[-1]*(self.dim/4)**2)
        if self.out_activation:
            self.sequential.add_module(self.out_activation, act_modules[self.out_activation])
        if self.spectral:
            self.apply(t.spectral_norm)
        self.optimizer = optim.Adadelta(self.parameters(), rho=0.95, eps=1e-7)

def get_d_block(con_in, con_out, activations, norm_layer=None):
    if norm_layer:
        return nn.Sequential(
            nn.ConvTranspose2d(con_in, con_out, 5, 1, 2),
            norm_layers[norm_layer](con_out),
            act_modules[activations],
            nn.Upsample(scale_factor=2, mode='nearest')
        )
    else:
        return nn.Sequential(
            nn.ConvTranspose2d(con_in, con_out, 5, 1, 2),
            act_modules[activations],
            nn.Upsample(scale_factor=2, mode='nearest')
        )

class DCNN(Base):
    def set_params(self, model_params):
        self.add_default_params({
            'cons':[1,8,16],
            'dim':32,
            'dim_x':None, 
            'dim_y':None, 
            'layer_units':[], 
            'activations':'relu', 
            'out_activation':None, 
            'dropouts':[0.0, 0.0], 
            'loss_func':'mee', 
            'spectral':False,
            'pooling':'max',
            'norm_layer':None,
            'avgpool':False,
        })
        super().set_params(model_params)

    def build_model(self):
        self.dim_co = int(self.dim/(2**(len(self.cons)-1)))
        self.cons.reverse()
        self.layer_units.reverse()
        if self.dim_x:
            self.layer_units.insert(0, self.dim_x)
        self.layer_units.append(self.cons[0] * self.dim_co * self.dim_co)
        self.loss_funcs['loss'] = loss_funcs[self.loss_func]

        self.sequential = nn.Sequential()
        if self.dim_x:
            for i in range(len(self.layer_units)-1):
                self.sequential.add_module('layer%d'%i, nn.Linear(self.layer_units[i], self.layer_units[i+1]))
                self.sequential.add_module('%s%d'%(self.activations, i), act_modules[self.activations])
        self.sequential.add_module('view', View(-1, self.cons[0], self.dim_co, self.dim_co))
        for i, con_in, con_out in zip(range(len(self.cons)-1), self.cons[:-1], self.cons[1:]):
            self.sequential.add_module('deconv%d'%i, get_d_block(con_in, con_out, self.activations, self.norm_layer))
        if self.dim_y:
            self.sequential.add_module('flatten', torch.nn.modules.Flatten())
            self.sequential.add_module('reshape', nn.Linear(self.dim*self.dim, self.dim_y))
        
        self.optimizer = optim.Adam(self.parameters(), lr=2.0e-4, betas=(0.5, 0.999))
