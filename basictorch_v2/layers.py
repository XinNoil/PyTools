import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Linear
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair

def softplus(x):
    """ Positivity constraint """
    softplus = torch.log(1+torch.exp(x))
    # Avoid infinities due to taking the exponent
    softplus = torch.where(softplus==float('inf'), x, softplus)
    return softplus

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

class Unsqueeze(nn.Module):
    def __init__(self, dim):
        self.dim = dim

    def forward(self, x):
        return torch.unsqueeze(x, self.dim)

class MLinear(nn.Module):
    def __init__(self, n, in_features: int, out_features: int, bias: bool = True,
                 device=None, dtype=None):
        super().__init__()
        self.linears = nn.ModuleList([Linear(in_features, out_features, bias, device, dtype) for i in range(n)])
    
    def forward(self, x):
        token_axis = len(list(x.shape))-2
        x = torch.swapaxes(x, 0, token_axis)
        x = torch.stack([linear(_x) for _x,linear in zip(x, self.linears)], dim=token_axis)
        return x

class PreNorm(nn.Module):
    def __init__(self, dim, fn):
        super().__init__()
        self.norm = nn.LayerNorm(dim)
        self.fn = fn

    def forward(self, x, **kwargs):
        return self.fn(self.norm(x), **kwargs)

class NormalGammaLinear(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features*4, bias)
        self.dim_y = out_features
    
    def forward(self, input):
        output = super().forward(input)
        gamma, logv, logalpha, logbeta = torch.split(output, [self.dim_y, self.dim_y, self.dim_y, self.dim_y], dim=-1)
        v = softplus(logv)
        alpha = softplus(logalpha)+1
        beta = softplus(logbeta)
        return torch.cat((gamma, v, alpha, beta), dim=-1)

class NormalGammaLinear_(Linear):
    def __init__(self, in_features, out_features, bias=True):
        super().__init__(in_features, out_features*3, bias)
        self.dim_y = out_features
    
    def forward(self, input):
        output = super().forward(input)
        logv, logalpha, logbeta = torch.split(output, [self.dim_y, self.dim_y, self.dim_y], dim=-1)
        v = softplus(logv)
        alpha = softplus(logalpha)+1
        beta = softplus(logbeta)
        return torch.cat((v, alpha, beta), dim=-1)

class GRL(Function):
    '''
    grl = GRL()
    output = grl(ouput)
    output.backward()
    '''
    def __init__(self, alpha = 1, *args, **kwargs):
        super().__init__(*args, **kwargs)
        self.alpha = alpha

    @property
    def alpha(self):
        return GRL._alpha
    
    @alpha.setter
    def alpha(self, num):
        GRL._alpha = num

    @staticmethod
    def forward(ctx,input):
        ctx.save_for_backward(input)
        return input
    
    @staticmethod
    def backward(ctx,grad_output):
        result, = ctx.saved_tensors
        return GRL._alpha*grad_output.neg()

def random_drop(x, p):
    m = torch.rand_like(x, dtype=torch.float16)
    x = x * (m>p)
    return x

class RandomDrop(nn.Module):
    def __init__(self, p, random_p=False):
        if p < 0 or p > 1:
            raise ValueError("Random drop probability has to be between 0 and 1, "
                             "but got {}".format(p))
        super().__init__()
        self.p = p
        self.random_p = random_p

    def forward(self, x):
        if self.training:
            return random_drop(x, torch.rand(x.shape[0], 1, device=x.device)*self.p if self.random_p else self.p)
        else:
            return x

acts = {
    'relu':torch.relu,
    'tanh':torch.tanh,
    'sigmoid':torch.sigmoid,
    'leakyrelu':nn.functional.leaky_relu,
    'elu':nn.functional.elu,
    'gelu':nn.functional.gelu,
    'softmax':torch.softmax,
}

act_modules = {
    'relu':nn.ReLU(),
    'tanh':nn.Tanh(),
    'sigmoid':nn.Sigmoid(),
    'leakyrelu':nn.modules.LeakyReLU(),
    'elu':nn.ELU(),
    'gelu':nn.GELU(),
    'softmax':nn.Softmax(-1),
}

acts = {
    'relu':nn.ReLU,
    'tanh':nn.Tanh,
    'sigmoid':nn.Sigmoid,
    'leakyrelu':nn.modules.LeakyReLU,
    'elu':nn.ELU,
    'gelu':nn.GELU,
    'softmax':nn.Softmax,
}

poolings={
    'max':nn.MaxPool2d,
    'avg':nn.AvgPool2d,
    None:nn.Identity
}

drops={
    'dropout':nn.Dropout,
    'featuredrop':RandomDrop,
    None:nn.Identity
}

norm_layers = {
    None: nn.Identity,
    'layernorm':nn.LayerNorm,
    'batchnorm':nn.BatchNorm2d,
}

Conv = {1:nn.Conv1d, 2:nn.Conv2d}
DeConv = {1:nn.ConvTranspose1d, 2:nn.ConvTranspose2d}
Maxpool = {1:nn.MaxPool1d, 2:nn.MaxPool2d}
UpSample = {1:nn.Upsample, 2:nn.Upsample}
BatchNorm = {1:nn.BatchNorm1d, 2:nn.BatchNorm2d}
AdaptiveAvgPool = {1:nn.AdaptiveAvgPool1d, 2:nn.AdaptiveAvgPool2d}