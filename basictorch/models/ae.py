from .base import *

class Encoder(nn.Module):
    def __init__(self, dim_x, dim_z, layer_units, p):
        super().__init__()
        self.p = p
        if self.p > 0:
            self.dropout = nn.Dropout(p)
        self.enc_layers = t.get_layers(dim_x, layer_units)
        self.z_layer = nn.Linear(layer_units[-1], dim_z)
    
    def forward(self, x):
        h = x
        if self.p > 0:
            h = self.dropout(h)
        for layer in self.enc_layers:
            h = torch.relu(layer(h))
        return self.z_layer(h)

class Decoder(nn.Module):
    def __init__(self, dim_z, dim_x, layer_units):
        super().__init__()
        layer_units_r = layer_units.copy()
        layer_units_r.reverse()
        self.dec_layers = t.get_layers(dim_z, layer_units_r)
        self.x_rec_layer = nn.Linear(layer_units_r[-1], dim_x)
    
    def forward(self, z):
        h = z
        for layer in self.dec_layers:
            h = torch.relu(layer(h))
        return self.x_rec_layer(h)

class AE(Base):
    def __init__(self, name, args, set_model_params=True, dim_x=None, dim_z=8, layer_units=[256,128,32], dropouts=[0.0, 0.0], loss_func='mrl', lr=2.0e-4, betas=(0.5, 0.999)):
        super(AE, self).__init__(name, args, set_model_params=True, dim_x=None, dim_z=8, layer_units=[256,128,32], dropouts=[0.0, 0.0], loss_func='mrl', lr=2.0e-4, betas=(0.5, 0.999)) if set_model_params else super(AE, self).__init__(name, args)
            
    def set_model_params(self, model_params):
        self.model_params = model_params['model_params']
        self.dim_x = model_params['dim_x']
        self.dim_z = model_params['dim_z']
        self.layer_units = model_params['layer_units']
        self.dropouts = model_params['dropouts']
        self.build_model()
        self.loss_funcs['loss'] = loss_funcs[model_params['loss_func']]
        self.optimizer = optim.Adam(self.parameters(), lr=model_params['lr'], betas=model_params['betas'])
    
    def build_model(self):
        self.encoder = Encoder(self.dim_x, self.dim_z, self.layer_units, self.dropouts[0])
        self.decoder = Decoder(self.dim_z, self.dim_x, self.layer_units)
        
    def forward(self, x):
        z = self.encoder(x)
        x_rec = self.decoder(z)
        return z, x_rec
    
    def predict(self, x):
        with torch.no_grad():
            return self.forward(x)
    
    def get_losses(self, batch_data):
        inputs, labels = batch_data
        outputs = self(inputs)
        losses={}
        losses['loss'] = self.loss_funcs['loss'](outputs, inputs)
        return losses
