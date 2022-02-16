import torch as T
import numpy as np
from .layers import softplus
eps=1e-6

def euclidean_error(output, target):
    return T.sqrt(T.sum((output - target)**2+eps, dim=-1))

def mean_euclidean_error(output, target, mean=True):
    if mean:
        return T.mean(euclidean_error(output, target))
    else:
        return euclidean_error(output, target)

def mean_rec_loss(x, x_rec):
    return mean_squared_error(x, x_rec)

def gen_loss(x, x_rec, z_avg, z_log_var, alpha=1.0):
    x_rec_loss = T.mean((x - x_rec).pow(2), dim=-1)
    z_kl_loss = -0.5 * T.sum(1.0 + z_log_var - z_avg.pow(2) - z_log_var.exp(), dim=-1)
    return alpha*z_kl_loss + x_rec_loss

def mean_gen_loss(x, x_rec, z_avg, z_log_var, alpha=1.0):
    return T.mean(gen_loss(x, x_rec, z_avg, z_log_var, alpha))

def mean_rec_kl_loss(x, x_rec, z_avg, z_log_var):
    x_rec_loss = T.mean((x - x_rec).pow(2), dim=-1)
    z_kl_loss = -0.5 * T.sum(1.0 + z_log_var - z_avg.pow(2) - z_log_var.exp(), dim=-1)
    return T.mean(x_rec_loss), T.mean(z_kl_loss)

def likelihood(d, sigma):
    return - ( - 0.5 * np.log(2.*np.pi) - np.log(sigma) - 0.5 * (d**2)/(sigma**2))

def dis_loss(y, y_pred, sigma):
    return likelihood(T.sqrt(T.sum((y - y_pred)**2, dim=-1)+eps), sigma)

def mean_dis_loss(y, y_pred, sigma, mean=True):
    if mean:
        return T.mean(dis_loss(y, y_pred, sigma))
    else:
        return dis_loss(y, y_pred, sigma)

class Mean_dis_loss(object):
    def __init__(self, sigma=None) -> None:
        super().__init__()
        self.sigma = sigma
    
    def __call__(self, y, y_pred):
        return mean_dis_loss(y, y_pred, self.sigma)

def mean_squared_error(y, y_pred):
    return T.mean((y_pred-y)**2)

def weighted_mean_squared_error(y, y_pred, w):
    return T.mean(w*(y_pred-y)**2)

def normpdf(d, sigma, B=np):
    return B.exp( - 0.5 * np.log(2.*np.pi) - np.log(sigma) - 0.5 * (d**2)/(sigma**2))

def root_mean_square_error(y, y_pred):
    return T.sqrt(mean_squared_error(y, y_pred))

def mean_root_mean_square_error(y, y_pred):
    return T.mean(T.sqrt(T.mean((y_pred-y)**2, dim=-1)))

# [pytorch-pne](https://github.com/github-jnauta/pytorch-pne/blob/master/models/pnn.py)
def negative_log_likelihood(outputs, truth, distance=T.sub):
    """ Compute the Negative Log Likelihood """
    mean, var = outputs
    diff = distance(truth, mean)
    var = softplus(var)
    # TODO: Check min and max variance
    loss = T.mean(T.div(diff**2, var))
    loss += T.mean(T.log(var))
    return loss

def nig_nll(y, gamma, v, alpha, beta):
    twoBlambda = 2*beta*(1+v)
    nll = 0.5*T.log(np.pi/v)  \
        - alpha*T.log(twoBlambda)  \
        + (alpha+0.5) * T.log(v*(y-gamma)**2 + twoBlambda)  \
        + T.lgamma(alpha)  \
        - T.lgamma(alpha+0.5)
    return T.mean(nll)

def nig_reg(y, gamma, v, alpha):
    evi = 2*v+(alpha)
    reg = T.abs(y-gamma)*evi
    return T.mean(reg)

def triee(output, label): # loss = sqrt( a^2+b^2-2*a*b*cos(theta))
    return T.sqrt(output[:,0]**2+label[:,0]**2-2*output[:,0]*label[:,0]*T.cos(output[:,1]-label[:,1]))

def trimee(output, label):
    return T.mean(triee(output, label))

def imuloss(output, label, alpha=0.1):
    dif = T.abs(output-label)
    return T.mean(dif[:,0] + alpha*dif[:,1]) #  # alpha*dif[:,1]

def imuloss_l(output, label, alpha=1.0):
    dif = T.abs(output-label)
    return T.mean(dif[:,0]) #  # alpha*dif[:,1]

def imuloss_psi(output, label, alpha=1.0):
    dif = T.abs(output-label)
    return T.mean(dif[:,1]) # dif[:,0] # alpha*

def identity(labels, outputs):
    return outputs

ee = euclidean_error
mee = mean_euclidean_error
mrl = mean_rec_loss
mgl = mean_gen_loss
mdl = mean_dis_loss
mse = mean_squared_error
nll = negative_log_likelihood
rmse = root_mean_square_error
mrmse = mean_root_mean_square_error
wmse = weighted_mean_squared_error
loss_funcs={
    'identity':identity,
    'ee':ee,
    'mee':mee,
    'mrl':mrl,
    'mgl':mgl,
    'mdl':mdl,
    'mse':mse,
    'nll':nll,
    'rmse':rmse,
    'mrmse':mrmse,
    'likelihood':likelihood,
    'trimee':trimee,
    'imuloss':imuloss,
    'imuloss_l':imuloss_l,
    'imuloss_psi':imuloss_psi,
    'mrkl':mean_rec_kl_loss,
    'wmse':wmse,
    'crossentropy':T.nn.CrossEntropyLoss()
}

def get_loss_func(loss_func, **params):
    if loss_func =='mdl':
        return Mean_dis_loss(**params)
    else:
        return loss_funcs[loss_func]

# BCELoss = T.nn.BCELoss()

def js_adv_loss(p, v):
    return T.mean(-(v*T.log(p)+(1-v)*T.log(1-p)))

def ls_adv_loss(p, v):
    return T.mean((p-v) ** 2)

def w_adv_loss(p, v):
    if v>0.5:
        return - p.mean()
    else:
        return p.mean()

adv_losses={
    'js':js_adv_loss,
    'ls':ls_adv_loss,
    'w':w_adv_loss,
}