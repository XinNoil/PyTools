import torch
from torch import nn
import torch.nn.functional as F
from torch.autograd import Function
from torch.nn import Linear
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair
import gpytorch
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MultitaskKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.models import AbstractVariationalGP, GP
from gpytorch.models.deep_gps import AbstractDeepGPLayer, AbstractDeepGP, DeepLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood

def softplus(x):
    """ Positivity constraint """
    softplus = torch.log(1+torch.exp(x))
    # Avoid infinities due to taking the exponent
    softplus = torch.where(softplus==float('inf'), x, softplus)
    return softplus

class E(nn.Module):
    def forward(self, x):
        return x

class View(nn.Module):
    def __init__(self, *shape):
        super().__init__()
        self.shape = shape

    def forward(self, input):
        return input.view(*self.shape)

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

class DeepGPLayer(AbstractDeepGPLayer):
    def __init__(self, input_dims, output_dims, inducing_points):
        self.output_dims = output_dims

        variational_distribution = CholeskyVariationalDistribution(
            num_inducing_points=inducing_points.shape[-2],
            batch_shape=torch.Size([output_dims]) if output_dims is not None else torch.Size([])
        )

        variational_strategy = VariationalStrategy(
            self,
            inducing_points,
            variational_distribution,
            learn_inducing_locations=True
        )

        super(DeepGPLayer, self).__init__(variational_strategy, input_dims, output_dims)

        self.mean_module = ConstantMean(batch_size=output_dims)
        self.covar_module = ScaleKernel(
            RBFKernel(batch_size=output_dims, ard_num_dims=input_dims),
            batch_size=output_dims, ard_num_dims=None
        )

    def forward(self, x):
        mean_x = self.mean_module(x)
        covar_x = self.covar_module(x)
        return MultivariateNormal(mean_x, covar_x)
    
    def __call__(self, x, *other_inputs, **kwargs):
        """
        Overriding __call__ isn't strictly necessary, but it lets us add concatenation based skip connections
        easily. For example, hidden_layer2(hidden_layer1_outputs, inputs) will pass the concatenation of the first
        hidden layer's outputs and the input data to hidden_layer2.
        """
        if len(other_inputs):
            if isinstance(x, MultitaskMultivariateNormal):
                x = x.rsample()

            processed_inputs = [
                inp.unsqueeze(0).expand(self.num_samples, *inp.shape)
                for inp in other_inputs
            ]

            x = torch.cat([x] + processed_inputs, dim=-1)

        return super().__call__(x, are_samples=bool(len(other_inputs)))

class DeepGP(AbstractDeepGP):
    def __init__(self, hidden_layer, dim_y, dim_h, layer_units, inducing_points=None, num_inducing=512):
        super(DeepGP, self).__init__()
        self.hidden_layer = hidden_layer
        if inducing_points is None:
            if (dim_y is None) or (dim_y == 1):
                inducing_points = torch.rand(num_inducing, dim_h)
            else:
                inducing_points = torch.rand(dim_y, num_inducing, dim_h)
            inducing_points = inducing_points * 2 - 1
        elif inducing_points.shape[0]>num_inducing:
            inducing_points = inducing_points[:num_inducing,:]

        self.last_layer = DeepGPLayer(
            input_dims=dim_h,
            output_dims=dim_y if dim_y>1 else None,
            inducing_points = inducing_points
        )
        
        if dim_y>1:
            self.likelihood = DeepLikelihood(MultitaskGaussianLikelihood(num_tasks=dim_y))
        else:
            self.likelihood = DeepLikelihood(GaussianLikelihood())
    
    def forward(self, x):
        for layer in self.hidden_layer[:-1]:
            x = torch.relu(layer(x))
        x = torch.tanh(self.hidden_layer[-1](x))
        o = self.last_layer(x)
        return o
    
    def predict(self, x):
        with gpytorch.settings.fast_computations(log_prob=False, solves=False), torch.no_grad():
            return self(x)

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
        return random_drop(x, torch.rand(x.shape[0], 1, device=x.device)*self.p if self.random_p else self.p)