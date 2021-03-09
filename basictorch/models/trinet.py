from .base import *

default_model_params={
    "loss_func":'mee', 
    "K":9,
    "threshold_tri":1,
    "threshold_self":1,
    "check_stable":False,
}

class TriModel(Base):
    def set_model_params(self, model_params):
        super().set_model_params(model_params, default_model_params)
        self.loss_funcs['loss'] = loss_funcs[self.loss_func]

    def build_model(self, share_model, sub_models, aug_model=None):
        self.share_model = share_model
        self.sub_models  = sub_models
        self.aug_model   = aug_model
    
    def initialize_model(self):
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

    def get_losses(self, batch_data):
        inputs, labels = batch_data
        if self.training:
            if self.aug_model:
                with torch.no_grad():
                    tri_inputs = [inputs, self.aug_model.decode_y(labels, random_z=True), self.aug_model.generate(labels, random_z=True)]
            else:
                tri_inputs = [inputs]*3
        else:
            tri_inputs = [inputs]*3
        tri_labels = [labels]*3
        tri_outputs = self(tri_inputs, tri=True)
        outputs = t.stack_mean(tri_outputs)
        return {'loss':t.stack_mean([loss_funcs['mee'](label, output) for label, output in zip(tri_labels, tri_outputs)]), 'mee':loss_funcs['mee'](labels, outputs)}

    def pseudo_labeling(self, x_u):
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
            y_s = torch.stack([torch.stack(self([x_u]*3, tri=True)) for i in range(self.K)])
            dis_self = torch.stack([torch.stack([loss_funcs['ee'](y_t, y_i) for y_t, y_i in zip(y_ts, y_k)]) for y_k in y_s])
            dis_self = torch.mean(dis_self, dim=0)
            stable = dis_self<self.threshold_self
            pseudo_0 = pseudo_0&stable[1]&stable[2]
            pseudo_1 = pseudo_1&stable[0]&stable[2]
            pseudo_2 = pseudo_2&stable[0]&stable[1]
        pseudo_labels = [t.stack_mean([y_ts[1], y_ts[2]])[pseudos[0]], t.stack_mean([y_ts[0], y_ts[2]])[pseudos[1]], t.stack_mean([y_ts[1], y_ts[1]])[pseudos[2]]]
        pseudo_losses = [loss_funcs['mee'](y_t[pseudo], pseudo_label) if torch.sum(pseudo)>0 else t.tensor(0) for y_t,pseudo,pseudo_label in zip(y_ts, pseudos, pseudo_labels)]
        return t.stack_mean(pseudo_losses)