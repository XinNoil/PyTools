import torch as T
import numpy as np

def euclidean_error(output, target):
    return T.sqrt(T.sum((output - target)**2, dim=-1))

def mean_euclidean_error(output, target):
    return T.mean(euclidean_error(output, target))

def mean_rec_loss(x, x_rec):
    x_rec_loss = T.mean((x - x_rec).pow(2), dim=-1)
    return T.mean(x_rec_loss)

def gen_loss(x, x_rec, z_avg, z_log_var):
    x_rec_loss = T.mean((x - x_rec).pow(2), dim=-1)
    z_kl_loss = -0.5 * T.sum(1.0 + z_log_var - z_avg.pow(2) - z_log_var.exp(), dim=-1)
    return z_kl_loss + x_rec_loss

def mean_gen_loss(x, x_rec, z_avg, z_log_var):
    return T.mean(gen_loss(x, x_rec, z_avg, z_log_var))

def dis_loss(y, y_pred, sigma):
    dis = T.sqrt(T.sum((y - y_pred)**2, dim=-1))
    return - ( - 0.5 * np.log(2.*np.pi) - np.log(sigma) - 0.5 * (dis**2)/(sigma**2))

def mean_dis_loss(y, y_pred, sigma):
    return T.mean(dis_loss(y, y_pred, sigma))

def mean_squared_error(y, y_pred):
    return T.mean((y_pred-y)**2)

def root_mean_square_error(y, y_pred):
    return T.sqrt(mean_squared_error(y, y_pred))

mee = mean_euclidean_error
mrl = mean_rec_loss
mgl = mean_gen_loss
mdl = mean_dis_loss
mse = mean_squared_error
rmse = root_mean_square_error

loss_funcs={
    'mee':mee,
    'mrl':mrl,
    'mgl':mgl,
    'mdl':mdl,
    'mse':mse,
    'rmse':rmse,
}

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