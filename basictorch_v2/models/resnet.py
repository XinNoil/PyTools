import pdb
import torch
from torch import Tensor
import torch.nn as nn
from ..layers import acts, BatchNorm, Conv, DeConv, Maxpool, UpSample, BatchNorm, AdaptiveAvgPool
from typing import Type, Any, Callable, Union, List, Optional
# import torchvision.models

def conv3x(in_planes, out_planes, stride = 1, groups = 1, dilation = 1, conv = None):
    """3x3 convolution with padding"""
    return conv(in_planes, out_planes, kernel_size=3, stride=stride,
                     padding=dilation, groups=groups, bias=False, dilation=dilation)

def conv1x(in_planes, out_planes, stride = 1, conv = None):
    """1x1 convolution"""
    return conv(in_planes, out_planes, kernel_size=1, stride=stride, bias=False)

def conv_block(conv_dim, inchannels, inplanes, kernel_size=7, stride=2, padding=3, bias=False, act='relu', pool_kernel_size=3, pool_stride=2, pool_padding=1) -> nn.Sequential:
    return nn.Sequential(
            # inchannels: 通道数，inplanes: filter个数, kernel_size: 卷积核尺寸，stride: 步长
            Conv[conv_dim](inchannels, inplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            BatchNorm[conv_dim](inplanes),
            acts[act](inplace=True),
            Maxpool[conv_dim](kernel_size=pool_kernel_size, stride=pool_stride, padding=pool_padding) if pool_kernel_size>1 else nn.Identity()
        )

def deconv_block(conv_dim, inchannels, inplanes, kernel_size=7, stride=2, padding=3, bias=False, act='relu', scale_factor=2) -> nn.Sequential:
    return nn.Sequential(
            DeConv[conv_dim](inchannels, inplanes, kernel_size=kernel_size, stride=stride, padding=padding, bias=bias),
            BatchNorm[conv_dim](inplanes),
            acts[act](inplace=True),
            UpSample[conv_dim](scale_factor=scale_factor) if scale_factor>1 else nn.Identity()
        )

class BasicBlock(nn.Module):
    expansion: int = 1

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_dim: int = 2,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        act: str = 'relu',
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm[conv_dim]
        if groups != 1 or base_width != 64:
            raise ValueError('BasicBlock only supports groups=1 and base_width=64')
        if dilation > 1:
            raise NotImplementedError("Dilation > 1 not supported in BasicBlock")
        # Both self.conv1 and self.downsample layers downsample the input when stride != 1
        self.conv1 = conv3x(inplanes, planes, stride, conv=Conv[conv_dim])
        self.bn1 = norm_layer(planes)
        self.act = acts[act](inplace=True)
        self.conv2 = conv3x(planes, planes, conv=Conv[conv_dim])
        self.bn2 = norm_layer(planes)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x: Tensor) -> Tensor:
        identity = x

        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)

        if self.downsample is not None:
            identity = self.downsample(x)

        out += identity
        out = self.act(out)

        return out

class Bottleneck(nn.Module):
    __constants__ = ['downsample']

    def __init__(
        self,
        inplanes: int,
        planes: int,
        conv_dim: int = 2,
        stride: int = 1,
        downsample: Optional[nn.Module] = None,
        groups: int = 1,
        base_width: int = 64,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        act: str = 'relu',
        expansion: int = 4,
        de_conv: bool = False,
        pyramid: bool = False,
        outplanes: int = None,
        *args,
        **kwargs
    ) -> None:
        super().__init__()
        if norm_layer is None:
            norm_layer = BatchNorm[conv_dim]
        width = int(planes * (base_width / 64.)) * groups
        if pyramid:
            if de_conv:
                planes = [inplanes, int(width/expansion), int(width/(expansion**2)), width if outplanes is None else outplanes]
            else:
                planes = [inplanes, width, width*expansion, width*(expansion**2)]
        else:
            outplanes = int(width/expansion if de_conv else width * expansion) if outplanes is None else outplanes
            planes = [inplanes, width, width, outplanes]
        
        # Both self.conv2 and self.downsample layers downsample the input when stride != 1
        conv = DeConv[conv_dim] if de_conv else Conv[conv_dim]
        self.conv1 = conv1x(planes[0], planes[1], conv=conv)
        self.bn1 = norm_layer(planes[1])
        self.conv2 = conv3x(planes[1], planes[2], stride, groups, dilation, conv=conv)
        self.bn2 = norm_layer(planes[2])
        self.conv3 = conv1x(planes[2], planes[3], conv=conv)
        self.bn3 = norm_layer(planes[3])
        self.act = acts[act](inplace=True)
        self.downsample = downsample
        self.stride = stride

    def forward(self, x):
        identity = x
        out = self.conv1(x)
        out = self.bn1(out)
        out = self.act(out)

        out = self.conv2(out)
        out = self.bn2(out)
        out = self.act(out)

        out = self.conv3(out)
        out = self.bn3(out)

        if self.downsample is not None:
            identity = self.downsample(x)
        
        out += identity
        out = self.act(out)

        return out


block_dict = {
    'basicblock': BasicBlock,
    'bottleneck': Bottleneck
}

class ResNet(nn.Module):

    def __init__(
        self,
        block: str,
        layers: List[int],
        planes: List[int],
        stride: int = 2,
        conv_dim: int = 2,
        inchannels: int = 3,
        inplanes: int = 64,
        expansion: int = None,
        dim_y: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilations: Optional[List[bool]] = None,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        act: str = 'relu',
        avgpool: bool = True,
        flatten: bool = True,
        keep_shape: bool = False,
        pyramid: bool = False
    ) -> None:
        super(ResNet, self).__init__()

        if block == 'basicblock':
            self.expansion = 1
        elif block == 'bottleneck' and expansion is None:
            self.expansion = 4
        else:
            self.expansion = expansion

        block = block_dict[block]
        if norm_layer is None:
            norm_layer = BatchNorm[conv_dim]
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.dilation = dilation
        if replace_stride_with_dilations is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilations = [False] * (len(layers)-1)

        self.groups = groups
        self.base_width = width_per_group
        self.flatten = flatten
        
        if inchannels is not None and not pyramid:
            if keep_shape:
                self.block1 = conv_block(conv_dim, inchannels, self.inplanes, act=act, kernel_size=1, stride=1, padding=0, bias=False, pool_kernel_size=1, pool_stride=1, pool_padding=0)
            else:
                self.block1 = conv_block(conv_dim, inchannels, self.inplanes, act=act)
        else:
            self.block1 = nn.Identity()
            
        if len(planes)==1:
            blocks = self._make_layer(block, planes[0], layers[0],
                                        dilate=replace_stride_with_dilations[0], conv_dim=conv_dim, act=act, single_layer=True, pyramid=pyramid)
            outchannels = planes[0] * (self.expansion**layers[0])
        else:
            blocks = [self._make_layer(block, planes[0], layers[0], conv_dim=conv_dim, act=act)]
            for layer, plane, replace_stride_with_dilation in zip(layers[1:], planes[1:], replace_stride_with_dilations):
                blocks.append(self._make_layer(block, plane, layer, stride=stride,
                                        dilate=replace_stride_with_dilation, conv_dim=conv_dim, act=act))
            outchannels = planes[0] * (self.expansion**len(layers))
        self.blocks = nn.Sequential(*blocks)
        self.avgpool = AdaptiveAvgPool[conv_dim]((1,)*conv_dim) if avgpool else nn.Identity()
        self.fc = nn.Linear(outchannels, dim_y) if dim_y else nn.Identity()

        for m in self.modules():
            if isinstance(m, Conv[conv_dim]):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (BatchNorm[conv_dim], nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]

    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, conv_dim: int = 2, act: str= 'relu', single_layer = False, pyramid: bool = False) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        layers = []

        outplanes = planes * (self.expansion**2) if pyramid else planes * self.expansion
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                conv1x(self.inplanes, outplanes, stride, conv=Conv[conv_dim]),
                norm_layer(outplanes),
            )
        layers.append(block(self.inplanes, planes, conv_dim, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, act, self.expansion, pyramid=pyramid))
        downsample = None
        for _ in range(1, blocks):
            if single_layer:
                self.inplanes = outplanes
                planes = planes if pyramid else planes * self.expansion
                outplanes = planes * (self.expansion**2) if pyramid else planes * self.expansion
                if stride != 1 or self.inplanes != outplanes:
                    downsample = nn.Sequential(
                        conv1x(self.inplanes, outplanes, stride, conv=Conv[conv_dim]),
                        norm_layer(outplanes),
                    )
                layers.append(block(self.inplanes, planes, conv_dim, stride, downsample, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, act=act, expansion=self.expansion, pyramid=pyramid))
            else:
                layers.append(block(self.inplanes, planes, conv_dim, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, act=act, expansion=self.expansion, pyramid=pyramid))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.block1(x)
        x = self.blocks(x)
        x = self.avgpool(x)
        if self.flatten:
            x = torch.flatten(x, 1)
        x = self.fc(x)
        return x

    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)

class DeResNet(nn.Module):
    def __init__(
        self,
        block: str,
        layers: List[int],
        planes: List[int],
        stride: int = 2,
        conv_dim: int = 2,
        inchannels: int = 3,
        inplanes: int = 64,
        expansion: int = None,
        dim_y: int = 10,
        zero_init_residual: bool = False,
        groups: int = 1,
        width_per_group: int = 64,
        replace_stride_with_dilations: Optional[List[bool]] = None,
        dilation: int = 1,
        norm_layer: Optional[Callable[..., nn.Module]] = None,
        act: str = 'relu',
        avgpool: bool = True,
        flatten: bool = True,
        keep_shape: bool = False,
        pyramid = False,
        outplanes: int = None
    ) -> None:
        super(DeResNet, self).__init__()

        if block == 'basicblock':
            self.expansion = 1
        elif block == 'bottleneck' and expansion is None:
            self.expansion = 4
        else:
            self.expansion = expansion

        block = block_dict[block]
        if norm_layer is None:
            norm_layer = BatchNorm[conv_dim]
        self._norm_layer = norm_layer

        self.inplanes = inplanes
        self.dilation = dilation
        if replace_stride_with_dilations is None:
            # each element in the tuple indicates if we should replace
            # the 2x2 stride with a dilated convolution instead
            replace_stride_with_dilations = [False] * (len(layers)-1)

        self.groups = groups
        self.base_width = width_per_group
        self.flatten = flatten
            
        if len(planes)==1:
            blocks = self._make_layer(block, planes[0], layers[0],
                                        dilate=replace_stride_with_dilations[0], conv_dim=conv_dim, act=act, single_layer=True, pyramid=pyramid, _outplanes=outplanes)
            outchannels = planes[0] / (self.expansion**layers[0])
        else:
            blocks = [self._make_layer(block, planes[0], layers[0], conv_dim=conv_dim, act=act)]
            for layer, plane, replace_stride_with_dilation in zip(layers[1:], planes[1:], replace_stride_with_dilations):
                blocks.append(self._make_layer(block, plane, layer, stride=stride,
                                        dilate=replace_stride_with_dilation, conv_dim=conv_dim, act=act))
            outchannels = planes[0] / (self.expansion**len(layers))
        
        if inchannels is not None and not pyramid:
            if keep_shape:
                self.block1 = deconv_block(conv_dim, self.inplanes, inchannels, act=act, kernel_size=1, stride=1, padding=0, bias=False, scale_factor=1)
            else:
                self.block1 = deconv_block(conv_dim, self.inplanes, inchannels, act=act)
        else:
            self.block1 = nn.Identity()
        self.blocks = nn.Sequential(*blocks)
        self.avgpool = AdaptiveAvgPool[conv_dim]((1,)*conv_dim) if avgpool else nn.Identity()
        self.fc = nn.Linear(outchannels, dim_y) if dim_y else nn.Identity()
        
        for m in self.modules():
            if isinstance(m, Conv[conv_dim]):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
            elif isinstance(m, (BatchNorm[conv_dim], nn.GroupNorm)):
                nn.init.constant_(m.weight, 1)
                nn.init.constant_(m.bias, 0)

        # Zero-initialize the last BN in each residual branch,
        # so that the residual branch starts with zeros, and each residual block behaves like an identity.
        # This improves the model by 0.2~0.3% according to https://arxiv.org/abs/1706.02677
        if zero_init_residual:
            for m in self.modules():
                if isinstance(m, Bottleneck):
                    nn.init.constant_(m.bn3.weight, 0)  # type: ignore[arg-type]
                elif isinstance(m, BasicBlock):
                    nn.init.constant_(m.bn2.weight, 0)  # type: ignore[arg-type]
    
    def _make_layer(self, block: Type[Union[BasicBlock, Bottleneck]], planes: int, blocks: int,
                    stride: int = 1, dilate: bool = False, conv_dim: int = 2, act: str= 'relu', single_layer: bool = False, pyramid: bool = False, _outplanes=None) -> nn.Sequential:
        norm_layer = self._norm_layer
        downsample = None
        previous_dilation = self.dilation
        if dilate:
            self.dilation *= stride
            stride = 1
        layers = []
        
        outplanes = (planes*(self.expansion**2) if pyramid else planes*self.expansion) if blocks>1 else _outplanes
        if stride != 1 or self.inplanes != outplanes:
            downsample = nn.Sequential(
                conv1x(self.inplanes, outplanes, stride, conv=Conv[conv_dim]),
                norm_layer(outplanes),
            )
        layers.append(block(self.inplanes, planes*4, conv_dim, stride, downsample, self.groups,
                            self.base_width, previous_dilation, norm_layer, act, self.expansion, True, pyramid=pyramid, outplanes=outplanes))
        
        downsample = None
        for _ in range(1, blocks):
            if single_layer:
                self.inplanes = outplanes
                planes = planes if pyramid else int(planes / self.expansion)
                if _ < blocks-1 or _outplanes is None:
                    outplanes = planes*(self.expansion**2) if pyramid else planes*self.expansion
                else:
                    outplanes = _outplanes
                if stride != 1 or self.inplanes != outplanes:
                    downsample = nn.Sequential(
                        conv1x(self.inplanes, outplanes, stride, conv=Conv[conv_dim]),
                        norm_layer(outplanes),
                    )
                layers.append(block(self.inplanes, planes*4, conv_dim, stride, downsample, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, act=act, expansion=self.expansion, de_conv=True, pyramid=pyramid, outplanes=outplanes))
                
            else:
                layers.append(block(self.inplanes, planes, conv_dim, groups=self.groups,
                                base_width=self.base_width, dilation=self.dilation,
                                norm_layer=norm_layer, act=act, expansion=self.expansion, de_conv=True))

        return nn.Sequential(*layers)

    def _forward_impl(self, x: Tensor) -> Tensor:
        # See note [TorchScript super()]
        x = self.blocks(x)
        x = self.block1(x)
        x = self.avgpool(x)
        if self.flatten:
            x = torch.flatten(x, 1)
        x = self.fc(x)
        return x
    
    def forward(self, x: Tensor) -> Tensor:
        return self._forward_impl(x)
