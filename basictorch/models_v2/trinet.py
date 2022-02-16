from .base import *

class TriModel(Base):
    def set_args_params(self):
        self._set_args_params(['check_stable','threshold_tri','threshold_self','alpha_tri','stable_K'])
        super().set_args_params()
    
    def set_params(self, model_params):
        self.add_default_params({
            'loss_func':'mee',
            'check_stable':False, 
            'threshold_tri':1.0,
            'threshold_self':1.0,
            'alpha_tri':1.0,
            'stable_K':9,
            'init_share':True,
            'aug_num':1,
            'aug_nograd':True,
            'augrd':False,
            'auge':1,
            'auges':0,
            'aug_intensity':0.5,
        })
        self.add_args_params([
            'reg_num',
            'model_type',
            # mdrop params
            'drop_consistency',
            'random_drop_p', # drop probability
            'random_p',
            'alpha_r', # weight of consistency loss
            # loccon params
            'identical_consistency', 
            # tri-net params
            'share_layer_units','reg_layer_units','dim_x_share','pseudo_label','share_dropouts','reg_dropouts','aug_model','aug_num','aug_nograd','aug_randn','aug_epoch','aug_epochs','aug_intensity',
        ])
        super().set_params(model_params)

    def build_model(self, share_model, sub_models, aug_model=None):
        self.share_model = share_model
        self.sub_models  = sub_models
        self.aug_model   = aug_model
    
    def initialize_model(self):
        if self.init_share:
            self.share_model.initialize_model()
        for sub_model in self.sub_models:
            sub_model.initialize_model()

    def forward(self, x, tri=False):
        if tri:
            z_list = [self.share_model(x_) for x_ in x]
            return [sub_model(z) for z,sub_model in zip(z_list, self.sub_models)]
        else:
            z = self.share_model(x)
            return t.stack_mean([sub_model(z) for sub_model in self.sub_models])
    
    def get_tri_inputs(self, inputs, labels):
        if (self.epoch>self.aug_epochs) & (self.epoch % self.aug_epoch==0):
            if self.aug_num == 1:
                if self.aug_randn:
                    tri_inputs = [inputs]*2
                    tri_inputs.insert(self.b%3, inputs+self.aug_intensity*torch.randn_like(inputs))
                else:
                    tri_inputs = [inputs]*2
                    x_p = self.generator.decode_y(labels, intensity=self.aug_intensity)
                    tri_inputs.insert(self.b%3, x_p.detach() if self.aug_nograd else x_p)
            elif self.aug_num == 2:
                if self.aug_randn:
                    tri_inputs = [inputs+self.aug_intensity*torch.randn_like(inputs) for i in range(2)]
                    tri_inputs.insert(self.b%3, inputs)
                else:
                    tri_inputs = [self.generator.decode_y(labels, intensity=self.aug_intensity) for i in range(2)]
                    if self.aug_nograd:
                        tri_inputs = [inputs.detach() for inputs in tri_inputs]
                    tri_inputs.insert(self.b%3, inputs)
        else:
            tri_inputs = [inputs]*3

    def get_losses(self, batch_data, loss_func=None):
        if loss_func is None:
            loss_func = self.loss_funcs['loss']
        inputs, labels = batch_data
        tri_inputs = self.get_tri_inputs(inputs, labels)
        tri_outputs = self(tri_inputs, tri=True)
        return {'loss':t.stack_mean([loss_func(labels, outputs) for outputs in tri_outputs])}

    def pseudo_labeling(self, x_u, batch_avg=False):
        self.train_mode(False)
        y_ts = self([x_u]*3, tri=True)
        self.train_mode(True)
        dis_0_1 = loss_funcs['ee'](y_ts[1], y_ts[0])
        dis_0_2 = loss_funcs['ee'](y_ts[2], y_ts[0])
        dis_1_2 = loss_funcs['ee'](y_ts[1], y_ts[2])
        pseudo_0 = (dis_1_2<self.threshold_tri)&(dis_0_2>self.alpha_tri*self.threshold_tri)&(dis_0_1>self.alpha_tri*self.threshold_tri)
        pseudo_1 = (dis_0_2<self.threshold_tri)&(dis_0_1>self.alpha_tri*self.threshold_tri)&(dis_1_2>self.alpha_tri*self.threshold_tri)
        pseudo_2 = (dis_0_1<self.threshold_tri)&(dis_0_2>self.alpha_tri*self.threshold_tri)&(dis_1_2>self.alpha_tri*self.threshold_tri)
        pseudos = torch.stack([pseudo_0, pseudo_1, pseudo_2])
        if self.check_stable:
            y_s = torch.stack([torch.stack(self([x_u]*3, tri=True)) for i in range(self.stable_K)])
            dis_self = torch.stack([torch.stack([loss_funcs['ee'](y_t, y_i) for y_t, y_i in zip(y_ts, y_k)]) for y_k in y_s])
            dis_self = torch.mean(dis_self, dim=0)
            stable = dis_self<self.threshold_self
            pseudo_0 = pseudo_0&stable[1]&stable[2]
            pseudo_1 = pseudo_1&stable[0]&stable[2]
            pseudo_2 = pseudo_2&stable[0]&stable[1]
        pseudo_labels = [t.stack_mean([y_ts[1], y_ts[2]])[pseudos[0]], t.stack_mean([y_ts[0], y_ts[2]])[pseudos[1]], t.stack_mean([y_ts[1], y_ts[1]])[pseudos[2]]]
        pseudo_losses = [loss_funcs['mee'](y_t[pseudo], pseudo_label) if torch.sum(pseudo)>0 else t.tensor(0) for y_t,pseudo,pseudo_label in zip(y_ts, pseudos, pseudo_labels)]
        if batch_avg:
            pseudo_losses = [(pseudo_loss*pseudo_label.shape[0])/x_u.shape[0] for pseudo_loss,pseudo_label in zip(pseudo_losses,pseudo_labels)]
        return t.stack_mean(pseudo_losses)