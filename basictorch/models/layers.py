import torch
import gpytorch
from torch.utils.data import TensorDataset, DataLoader
from gpytorch.means import ConstantMean, MultitaskMean
from gpytorch.kernels import RBFKernel, ScaleKernel, MultitaskKernel
from gpytorch.variational import VariationalStrategy, CholeskyVariationalDistribution
from gpytorch.distributions import MultitaskMultivariateNormal, MultivariateNormal
from gpytorch.models import AbstractVariationalGP, GP
from gpytorch.models.deep_gps import AbstractDeepGPLayer, AbstractDeepGP, DeepLikelihood
from gpytorch.likelihoods import MultitaskGaussianLikelihood, GaussianLikelihood

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

from torch.nn import Linear
import torch.nn.functional as F
from torch.nn.modules import conv
from torch.nn.modules.utils import _pair

#define _l2normalization
def _l2normalize(v, eps=1e-12):
    return v / (torch.norm(v) + eps)

def max_singular_value(W, u=None, Ip=1):
    """
    power iteration for weight parameter
    """
    #xp = W.data
    if not Ip >= 1:
        raise ValueError("Power iteration should be a positive integer")
    if u is None:
        u = torch.FloatTensor(1, W.size(0)).normal_(0, 1).cuda()
    _u = u
    for _ in range(Ip):
        _v = _l2normalize(torch.matmul(_u, W.data), eps=1e-12)
        _u = _l2normalize(torch.matmul(_v, torch.transpose(W.data, 0, 1)), eps=1e-12)
    sigma = torch.sum(F.linear(_u, torch.transpose(W.data, 0, 1)) * _v)
    return sigma, _u

class SNLinear(Linear):
    r"""Applies a linear transformation to the incoming data: :math:`y = Ax + b`
       Args:
           in_features: size of each input sample
           out_features: size of each output sample
           bias: If set to False, the layer will not learn an additive bias.
               Default: ``True``
       Shape:
           - Input: :math:`(N, *, in\_features)` where :math:`*` means any number of
             additional dimensions
           - Output: :math:`(N, *, out\_features)` where all but the last dimension
             are the same shape as the input.
       Attributes:
           weight: the learnable weights of the module of shape
               `(out_features x in_features)`
           bias:   the learnable bias of the module of shape `(out_features)`

           W(Tensor): spectrally normalized weight

           u (Tensor): the right largest singular value of W.
       """
    def __init__(self, in_features, out_features, bias=True):
        super(SNLinear, self).__init__(in_features, out_features, bias)
        self.register_buffer('u', torch.Tensor(1, out_features).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.linear(input, self.W_, self.bias)

class SNConv2d(conv._ConvNd):

    r"""Applies a 2D convolution over an input signal composed of several input
    planes.
    In the simplest case, the output value of the layer with input size
    :math:`(N, C_{in}, H, W)` and output :math:`(N, C_{out}, H_{out}, W_{out})`
    can be precisely described as:
    .. math::
        \begin{array}{ll}
        out(N_i, C_{out_j})  = bias(C_{out_j})
                       + \sum_{{k}=0}^{C_{in}-1} weight(C_{out_j}, k)  \star input(N_i, k)
        \end{array}
    where :math:`\star` is the valid 2D `cross-correlation`_ operator,
    :math:`N` is a batch size, :math:`C` denotes a number of channels,
    :math:`H` is a height of input planes in pixels, and :math:`W` is
    width in pixels.
    | :attr:`stride` controls the stride for the cross-correlation, a single
      number or a tuple.
    | :attr:`padding` controls the amount of implicit zero-paddings on both
    |  sides for :attr:`padding` number of points for each dimension.
    | :attr:`dilation` controls the spacing between the kernel points; also
      known as the à trous algorithm. It is harder to describe, but this `link`_
      has a nice visualization of what :attr:`dilation` does.
    | :attr:`groups` controls the connections between inputs and outputs.
      `in_channels` and `out_channels` must both be divisible by `groups`.
    |       At groups=1, all inputs are convolved to all outputs.
    |       At groups=2, the operation becomes equivalent to having two conv
                 layers side by side, each seeing half the input channels,
                 and producing half the output channels, and both subsequently
                 concatenated.
            At groups=`in_channels`, each input channel is convolved with its
                 own set of filters (of size `out_channels // in_channels`).
    The parameters :attr:`kernel_size`, :attr:`stride`, :attr:`padding`, :attr:`dilation` can either be:
        - a single ``int`` -- in which case the same value is used for the height and width dimension
        - a ``tuple`` of two ints -- in which case, the first `int` is used for the height dimension,
          and the second `int` for the width dimension
    .. note::
         Depending of the size of your kernel, several (of the last)
         columns of the input might be lost, because it is a valid `cross-correlation`_,
         and not a full `cross-correlation`_.
         It is up to the user to add proper padding.
    .. note::
         The configuration when `groups == in_channels` and `out_channels = K * in_channels`
         where `K` is a positive integer is termed in literature as depthwise convolution.
         In other words, for an input of size :math:`(N, C_{in}, H_{in}, W_{in})`, if you want a
         depthwise convolution with a depthwise multiplier `K`,
         then you use the constructor arguments
         :math:`(in\_channels=C_{in}, out\_channels=C_{in} * K, ..., groups=C_{in})`
    Args:
        in_channels (int): Number of channels in the input image
        out_channels (int): Number of channels produced by the convolution
        kernel_size (int or tuple): Size of the convolving kernel
        stride (int or tuple, optional): Stride of the convolution. Default: 1
        padding (int or tuple, optional): Zero-padding added to both sides of the input. Default: 0
        dilation (int or tuple, optional): Spacing between kernel elements. Default: 1
        groups (int, optional): Number of blocked connections from input channels to output channels. Default: 1
        bias (bool, optional): If ``True``, adds a learnable bias to the output. Default: ``True``
    Shape:
        - Input: :math:`(N, C_{in}, H_{in}, W_{in})`
        - Output: :math:`(N, C_{out}, H_{out}, W_{out})` where
          :math:`H_{out} = floor((H_{in}  + 2 * padding[0] - dilation[0] * (kernel\_size[0] - 1) - 1) / stride[0] + 1)`
          :math:`W_{out} = floor((W_{in}  + 2 * padding[1] - dilation[1] * (kernel\_size[1] - 1) - 1) / stride[1] + 1)`
    Attributes:
        weight (Tensor): the learnable weights of the module of shape
                         (out_channels, in_channels, kernel_size[0], kernel_size[1])
        bias (Tensor):   the learnable bias of the module of shape (out_channels)
        W(Tensor): spectrally normalized weight
        u (Tensor): the right largest singular value of W.
    .. _cross-correlation:
        https://en.wikipedia.org/wiki/Cross-correlation
    .. _link:
        https://github.com/vdumoulin/conv_arithmetic/blob/master/README.md
        https://github.com/godisboy/SN-GAN/blob/master/src/snlayers/snconv2d.py
    """
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True, padding_mode='zeros'):
        kernel_size = _pair(kernel_size)
        stride = _pair(stride)
        padding = _pair(padding)
        dilation = _pair(dilation)
        super(SNConv2d, self).__init__(
            in_channels, out_channels, kernel_size, stride, padding, dilation,
            False, _pair(0), groups, bias, padding_mode)
        self.register_buffer('u', torch.Tensor(1, out_channels).normal_())

    @property
    def W_(self):
        w_mat = self.weight.view(self.weight.size(0), -1)
        sigma, _u = max_singular_value(w_mat, self.u)
        self.u.copy_(_u)
        return self.weight / sigma

    def forward(self, input):
        return F.conv2d(input, self.W_, self.bias, self.stride,
                        self.padding, self.dilation, self.groups)