# Deep Evidential Regression
from .base import *
from .layers import NormalGammaLinear, NormalGammaLinear_
from .losses import nig_nll, nig_reg, euclidean_error
import mtools as mt

class Dense(nn.Module):
    def __init__(self, dim_x, dim_y, layer_units, activations, OutLayer):
        super().__init__()
        self.activations = activations
        self.layers = t.get_layers(dim_x, layer_units)
        self.out_layer = OutLayer(layer_units[-1], dim_y)
    
    def forward(self, x):
        for layer in self.layers:
            x = acts[self.activations](layer(x))
        return self.out_layer(x)

default_model_params={
    'dim_x':None, 
    'dim_y':None, 
    'layer_units':[256,128,32], 
    'activations':'relu', 
    'lam':1e-2,
    'epsilon':0,
    'split':True,
}

class ENN(Base): 
    def set_model_params(self, model_params):
        super().set_model_params(model_params, default_model_params)
        self.build_model()
        self.optimizer = optim.Adadelta(self.enn.parameters(), rho=0.95, eps=1e-7)
        if self.split:
            self.optimizer_D = optim.Adadelta(self.dnn.parameters(), rho=0.95, eps=1e-7)

    def build_model(self):
        if self.split:
            self.dnn = Dense(dim_x=self.dim_x, dim_y=self.dim_y, layer_units=self.layer_units, activations=self.activations, OutLayer=nn.Linear)
            self.enn = Dense(dim_x=self.dim_x, dim_y=self.dim_y, layer_units=self.layer_units, activations=self.activations, OutLayer=NormalGammaLinear_)
        else:
            self.enn = Dense(dim_x=self.dim_x, dim_y=self.dim_y, layer_units=self.layer_units, activations=self.activations, OutLayer=NormalGammaLinear)

    def forward(self, x):
        return torch.cat((self.dnn(x), self.enn(x)), dim=-1) if self.split else self.enn(x)
    
    def get_losses(self, batch_data):
        inputs, labels = batch_data
        outputs = self(inputs)
        gamma, v, alpha, beta = torch.split(outputs, [self.dim_y, self.dim_y, self.dim_y, self.dim_y], dim=-1)
        nll_loss = nig_nll(labels, gamma, v, alpha, beta)
        reg_loss = nig_reg(labels, gamma, v, alpha)
        loss = nll_loss + self.lam * reg_loss
        mee = loss_funcs['mee'](gamma, labels)
        return {'loss':loss, 'nll':nll_loss, 'reg':reg_loss, 'mee':mee}
    
    def train_on_batch(self, b, batch_data):
        if self.split:
            inputs, labels = batch_data
            self.optimizer_D.zero_grad()
            mean = self.dnn(inputs)
            loss = loss_funcs['mee'](mean, labels)
            loss.backward()
            self.optimizer_D.step()
            
            self.optimizer.zero_grad()
            outputs = self(inputs)
            gamma, v, alpha, beta = torch.split(outputs, [self.dim_y, self.dim_y, self.dim_y, self.dim_y], dim=-1)
            nll_loss = nig_nll(labels, gamma, v, alpha, beta)
            reg_loss = nig_reg(labels, gamma, v, alpha)
            loss = nll_loss + self.lam * reg_loss
            loss.backward()
            self.optimizer.step()
            return self.get_losses(batch_data)
        else:
            self.optimizer.zero_grad()
            losses = self.get_losses(batch_data)
            losses['loss'].backward()
            self.optimizer.step()
            return losses

    def train(self, batch_size=16, epochs=0, validation=True, reporters=['loss','mee','nll','reg','test_loss','test_mee'], monitor='mee', test_monitor=None, initialize=True):
        super().train(batch_size=batch_size, epochs=epochs, validation=validation, reporters=reporters, monitor=monitor, test_monitor=test_monitor, initialize=initialize)
    
    def save_end(self):
        super().save_end()
        self.save_output(*self.datasets.train_dataset.tensors, 'train_dataset_source')
        self.save_output(*self.datasets.test_dataset.tensors, 'test_dataset_source')
        self.save_output(*self.datasets.train_dataset_target.tensors, 'train_dataset_target')
        self.save_output(*self.datasets.test_dataset_target.tensors,  'test_dataset_target')
    
    def save_output(self, x, y, name):
        outputs = self(x)
        gamma, v, alpha, beta = torch.split(outputs, [self.dim_y, self.dim_y, self.dim_y, self.dim_y], dim=-1)
        err = euclidean_error(y, gamma)
        aleatoric = beta/(alpha-1)
        epistemic = beta/(v*(alpha-1))
        output_tensor = torch.cat((y, gamma, aleatoric, epistemic, err.view(-1, 1)), dim=-1)
        mt.csvwrite(t.get_filename(self.args, name,'csv'), t.t2n(output_tensor))
    