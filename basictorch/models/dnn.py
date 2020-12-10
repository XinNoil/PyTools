from .base import *

default_model_params={
    'dim_x':None, 
    'dim_y':None, 
    'layer_units':[256,128,32], 
    'activations':'relu', 
    'out_activation':None, 
    'dropouts':[0.0, 0.0], 
    'loss_func':'mee', 
    'spectral':False,
}

class DNN(Base):
    def set_model_params(self, model_params):
        super().set_model_params(model_params, default_model_params)
        self.layer_units.insert(0, self.dim_x)
        self.loss_funcs['loss'] = loss_funcs[self.loss_func]
        self.build_model()
        self.optimizer = optim.Adadelta(self.parameters(), rho=0.95, eps=1e-7)

    def build_model(self):
        self.sequential = nn.Sequential()
        if self.dropouts[0]>0:
            self.sequential.add_module('dropout_i', nn.Dropout(self.dropouts[0]))
        for l in range(len(self.layer_units)-1):
            self.sequential.add_module('layer%d'%l, nn.Linear(self.layer_units[l], self.layer_units[l+1]))
            self.sequential.add_module('%s%d'%(self.activations, l), act_modules[self.activations])
            if self.dropouts[1]>0:
                self.sequential.add_module('dropout_%d'%l, nn.Dropout(self.dropouts[1]))
        self.sequential.add_module('out_layer', nn.Linear(self.layer_units[-1], self.dim_y))
        if self.spectral:
            self.apply(t.spectral_norm)

def train_dnn(args, Ds, dnn=None, model_params={}):
    if not dnn:
        dnn = DNN
    for e in range(args.trails):
        args.exp_no = t.get_exp_no(args, e+1)
        model = dnn('dnn', args,
                dim_x=Ds.train_dataset.tensors[0].shape[1], 
                dim_y=Ds.train_dataset.tensors[1].shape[1],
                layer_units = args.layer_units,
                dropouts = args.dropouts,
                **model_params
            ).to(t.device)
        print(model)
        model.set_datasets(Ds)
        model.train(batch_size=args.batch_size, epochs=args.epochs, initialize=True)
    return model