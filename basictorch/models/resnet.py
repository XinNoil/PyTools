from basictorch.models.base import *
from .layers import View

# from torchvision.models import Bottleneck, BasicBlock, conv1x1, conv3x3

def conv3x3(in_planes, out_planes, stride=1, groups=1, dilation=1, Conv2d=nn.Conv2d):
    """3x3 convolution with padding"""
    return Conv2d(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)


def conv1x1(in_planes, out_planes, stride=1, Conv2d=nn.Conv2d):
    """1x1 convolution"""
    return Conv2d(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

class BasicBlock(nn.Module):
    expansion = 1
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, Conv2d=nn.Conv2d):
        super(BasicBlock, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x3(inplanes, planes, stride, Conv2d=Conv2d)
        self.bn1 = norm_layer(planes)
        self.relu = nn.ReLU(inplace=True)
        self.conv2 = conv3x3(planes, planes, Conv2d=Conv2d)
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out


class Bottleneck(nn.Module):
    expansion = 4
    __constants__ = ['downsample']

    def __init__(self, inplanes, planes, stride=1, downsample=None, groups=1,
                 base_width=64, dilation=1, norm_layer=None, Conv2d=nn.Conv2d):
        super(Bottleneck, self).__init__()
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        width = int(planes * (base_width / 64.)) * groups
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv1x1(inplanes, width, Conv2d=Conv2d)
        self.bn1 = norm_layer(width)
        self.conv2 = conv3x3(width, width, stride, groups, dilation, Conv2d=Conv2d)
        self.bn2 = norm_layer(width)
        self.conv3 = conv1x1(width, planes * self.expansion, Conv2d=Conv2d)
        self.bn3 = norm_layer(planes * self.expansion)
        self.relu = nn.ReLU(inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.relu(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.relu(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.relu(out)

        return out

default_model_params = {
    'cons':[1,8,16],
    'dim':64,
    'dim_x':None, 
    'dim_y':None, 
    'layer_units':[], 
    'out_activation':None, 
    'dropouts':[0.0, 0.0], 
    'loss_func':'mee', 
    'spectral':False,
    'block':'basicblock',
    'blocks':3,
    'zero_init_residual':False,
    'groups':1, 
    'replace_stride_with_dilation':False,
    'avgpool':True,
    'pooling':'max',
}

block_dict = {
    'basicblock': BasicBlock,
    'bottleneck': Bottleneck,
}

class ResNet(Base):
    def set_model_params(self, model_params):
        Base.set_model_params(self, model_params, default_model_params)
        self.block = block_dict[self.block]
        self.loss_funcs['loss'] = loss_funcs[self.loss_func]
        self.dilation = 1
        self.dim_co = int(self.dim / 2)
        self.inplanes = self.cons[1]
        self._norm_layer = nn.BatchNorm2d
        self.pooling = poolings[self.pooling]
        self.base_width = self.dim
        self.build_model()
        self.optimizer = optim.Adadelta(self.parameters(), rho=0.95, eps=1e-7)

    def build_model(self):
        self.sequential = nn.Sequential()
        if self.dim_x:
            self.sequential.add_module('reshape', nn.Linear(self.dim_x, self.cons[0]*self.dim*self.dim))
        self.sequential.add_module('view', View(-1, self.cons[0], self.dim, self.dim))
        self.sequential.add_module(
            'conv',
            nn.Sequential(
                nn.Conv2d(self.cons[0], self.cons[1], 5, 1, 2, bias=False),
                self._norm_layer(self.cons[1]),
                nn.ReLU(inplace=True),
                self.pooling,
                # nn.MaxPool2d(kernel_size=3, stride=2, padding=1),
            )
        )
        self.sequential.add_module(
            'blocks',
            self._make_layer(self.block, self.cons[2], self.blocks, dilate=self.replace_stride_with_dilation)
        )
        if self.avgpool:
            self.sequential.add_module('avgpool', nn.AdaptiveAvgPool2d((1, 1)))
            self.sequential.add_module('flatten', torch.nn.modules.Flatten())
            self.sequential.add_module('out_layer', nn.Linear(self.cons[2] * self.block.expansion, self.dim_y))
        else:
            self.sequential.add_module('flatten', torch.nn.modules.Flatten())
            self.sequential.add_module('out_layer', nn.Linear(self.cons[2] * self.block.expansion * self.dim_co * self.dim_co, self.dim_y))
        if self.out_activation:
            self.sequential.add_module(self.out_activation, act_modules[self.out_activation])
        if self.spectral:
            self.apply(t.spectral_norm)

    def forward(self, x):
        for m in self.sequential:
            x = m(x)
        return x

    def initialize_model(self):
        super().initialize_model()
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if self.zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer))

        return nn.Sequential(*layers)

default_model_params_d = {
    'cons':[1,8,16],
    'dim':64,
    'dim_x':None, 
    'dim_y':None, 
    'layer_units':[], 
    'out_activation':None, 
    'dropouts':[0.0, 0.0], 
    'loss_func':'mee', 
    'spectral':False,
    'block':'basicblock',
    'blocks':3,
    'zero_init_residual':False,
    'groups':1, 
    'replace_stride_with_dilation':False,
    'norm_layer':None, 
    'avgpool':False,
    'pooling':'max'
}

class DeResNet(Base):
    def set_model_params(self, model_params):
        Base.set_model_params(self, model_params, default_model_params_d)
        self.block = block_dict[self.block]
        self.loss_funcs['loss'] = loss_funcs[self.loss_func]
        self.dilation = 1
        self.cons.reverse()
        self.dim_co = int(self.dim/2)
        self.inplanes = self.cons[0]
        self._norm_layer = nn.BatchNorm2d
        self.pooling = poolings[self.pooling]
        self.base_width = self.dim
        self.build_model()
        self.optimizer = optim.Adadelta(self.parameters(), rho=0.95, eps=1e-7)

    def build_model(self):
        self.sequential = nn.Sequential()
        # if self.avgpool:
        self.sequential.add_module('reshape', nn.Linear(self.dim_x, self.cons[0]*self.block.expansion*self.dim_co*self.dim_co))
        self.sequential.add_module('view', View(-1, self.cons[0] * self.block.expansion, self.dim_co, self.dim_co))

        self.sequential.add_module(
            'blocks',
            self._make_layer(self.block, self.cons[1], self.blocks, dilate=self.replace_stride_with_dilation)
        )

        self.sequential.add_module(
                'conv',
                nn.Sequential(
                    nn.ConvTranspose2d(self.cons[1], self.cons[2], 5, 1, 2, bias=False),
                    self._norm_layer(self.cons[2]),
                    nn.ReLU(inplace=True),
                    nn.Upsample(scale_factor=2, mode='nearest')
                )
            )

        self.sequential.add_module('flatten', torch.nn.modules.Flatten())
        self.sequential.add_module('out_layer', nn.Linear(self.cons[2] * self.block.expansion * self.dim * self.dim, self.dim_y))

    def forward(self, x):
        for m in self.sequential:
            x = m(x)
        return x

    # def initialize_model(self):
    #     super().initialize_model()
    #     for m in self.modules():
    #         if isinstance(m, nn.Conv2d):
    #             nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    #         elif isinstance(m, (nn.BatchNorm2d, nn.GroupNorm)):
    #             nn.init.constant_(m.weight, 1)
    #             nn.init.constant_(m.bias, 0)

    #     # Zero-initialize the last BN in each residual branch,
    #     # so that the residual branch starts with zeros, and each residual block behaves like an identity.
    #     # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
    #     if self.zero_init_residual:
    #         for m in self.modules():
    #             if isinstance(m, Bottleneck):
    #                 nn.init.constant_(m.bn3.weight, 0)
    #             elif isinstance(m, BasicBlock):
    #                 nn.init.constant_(m.bn2.weight, 0)

    def _make_layer(self, block, planes, blocks, stride=1, dilate=False):
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        if stride != 1 or self.inplanes != planes * block.expansion:
            downsample = nn.Sequential(
                conv1x1(self.inplanes, planes * block.expansion, stride, Conv2d=nn.ConvTranspose2d),
                norm_layer(planes * block.expansion),
            )

        layers = []
        layers.append(block(self.inplanes, planes, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, Conv2d=nn.ConvTranspose2d))
        self.inplanes = planes * block.expansion
        for _ in range(1, blocks):
            layers.append(block(self.inplanes, planes, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, Conv2d=nn.ConvTranspose2d))

        return nn.Sequential(*layers)
