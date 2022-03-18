import torch
from torch import nn
from ..losses import loss_funcs
from ..tools import stack_mean

class TriModel(nn.Module):
    def __init__(self, name, share_model, sub_models, loss_func='mee', check_stable=True, threshold_tri=1.0, threshold_self=1.0, alpha_tri=1.0, stable_K=9):
        super().__init__()
        self.name = name
        self.share_model = share_model
        self.sub_models = sub_models
        self.loss_func = loss_func
        self.check_stable = check_stable
        self.threshold_tri = threshold_tri
        self.threshold_self = threshold_self
        self.alpha_tri = alpha_tri
        self.stable_K = stable_K

    def forward(self, x, tri=False, **kwargs):
        if tri:
            if type(x) == list:
                z_list = [self.share_model(x_) for x_ in x]
                return [sub_model(z) for z,sub_model in zip(z_list, self.sub_models)]
            else:
                z = self.share_model(x)
                return [sub_model(z) for sub_model in self.sub_models]
        else:
            z = self.share_model(x)
            return stack_mean([sub_model(z) for sub_model in self.sub_models])

    def get_losses(self, batch_data, loss_func='mee'):
        inputs, labels = batch_data
        tri_inputs = [inputs]*3
        tri_outputs = self(tri_inputs, tri=True)
        return {'loss':stack_mean([loss_funcs[loss_func](labels, outputs) for outputs in tri_outputs])}

    def pseudo_labeling(self, x_u, batch_avg=False):
        self.train(False)
        y_ts = self([x_u]*3, tri=True)
        self.train(True)
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
        pseudo_labels = [stack_mean([y_ts[1], y_ts[2]])[pseudos[0]], stack_mean([y_ts[0], y_ts[2]])[pseudos[1]], stack_mean([y_ts[1], y_ts[1]])[pseudos[2]]]
        pseudo_losses = [loss_funcs['mee'](y_t[pseudo], pseudo_label) if torch.sum(pseudo)>0 else torch.tensor(0, dtype=torch.float, device=x_u.device) for y_t,pseudo,pseudo_label in zip(y_ts, pseudos, pseudo_labels)]
        if batch_avg:
            pseudo_losses = [(pseudo_loss*pseudo_label.shape[0])/x_u.shape[0] for pseudo_loss,pseudo_label in zip(pseudo_losses,pseudo_labels)]
        return stack_mean(pseudo_losses)