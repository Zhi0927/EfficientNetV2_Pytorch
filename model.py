from collections import OrderedDict
from functools import partial
from typing import Callable, Optional

import torch.nn as nn
import torch
from torch import Tensor


if hasattr(nn, 'SiLU'):
    SiLU = nn.SiLU
    print("Support")
else:
    # For compatibility with old PyTorch versions
    print("Not Support, for compatibility with old PyTorch versions")
    class SiLU(nn.Module):
        def forward(self, x):
            return x * torch.sigmoid(x)
        
        

def drop_path(x, drop_prob: float = 0., training: bool = False):
    if drop_prob == 0. or not training:
        return x
    keep_prob = 1 - drop_prob
    shape = (x.shape[0],) + (1,) * (x.ndim - 1)  # work with diff dim tensors, not just 2D ConvNets
    random_tensor = keep_prob + torch.rand(shape, dtype = x.dtype, device = x.device)
    random_tensor.floor_()  # binarize
    output = x.div(keep_prob) * random_tensor
    return output
class DropPath(nn.Module):
    def __init__(self, drop_prob = None):
        super(DropPath, self).__init__()
        self.drop_prob = drop_prob
    
    def forward(self, x):
        #self.training: if set model.eval(), self.training = False
        return drop_path(x, self.drop_prob, self.training)  
    
    
class SqueezeExcite(nn.Module):
    def __init__(self,
                 input_channels: int,   # <------- MBConv Input channels.
                 expand_channels: int,  # <------- SE Inputchannels (DW Output channels).
                 se_ratio: float = 0.25):        
        super(SqueezeExcite, self).__init__()
        
        
        squeeze_channels = int(input_channels * se_ratio)
             
        self.avg_pool = nn.AdaptiveAvgPool2d(1)
        
        self.conv_reduce = nn.Conv2d(expand_channels, squeeze_channels, 1)
        self.act1 = SiLU()  # alias Swish
        
        self.conv_expand = nn.Conv2d(squeeze_channels, expand_channels, 1)
        self.act2 = nn.Sigmoid()

        
    def forward(self, x: Tensor) -> Tensor:
        #scale = x.mean((2, 3), keepdim=True) # Global polling
        scale = self.avg_pool(x)
        scale = self.conv_reduce(scale)      # FC1
        scale = self.act1(scale)             # Swish
        scale = self.conv_expand(scale)      # FC2
        scale = self.act2(scale)             # Sigmoid
        return scale * x
    
    
class ConvBNAct(nn.Module):
    def __init__(self,
                 in_channels: int,
                 out_channels: int,
                 kernel_size: int = 3,
                 stride: int = 1,
                 groups: int = 1,
                 norm_layer: Optional[Callable[..., nn.Module]] = None,
                 activation_layer: Optional[Callable[..., nn.Module]] = None):        
        super(ConvBNAct, self).__init__()

        padding = (kernel_size - 1) // 2        
        
        if norm_layer is None:
            norm_layer = nn.BatchNorm2d
        if activation_layer is None:
            activation_layer = SiLU

        self.conv = nn.Conv2d(in_channels  = in_channels,
                              out_channels = out_channels,
                              kernel_size  = kernel_size,
                              stride       = stride,
                              padding      = padding,
                              groups       = groups,
                              bias         = False)

        self.bn = norm_layer(out_channels)
        self.act = activation_layer()

    def forward(self, x):
        result = self.conv(x)
        result = self.bn(result)
        result = self.act(result)

        return result   


class MBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(MBConv, self).__init__()
       
        
        if stride not in [1, 2]:
            raise ValueError("illegal stride value.")
            
        self.out_channels = out_channels       
        self.drop_rate = drop_rate

        #shortcut used condition.(stride == 1, in-channels == out-channels)
        self.has_shortcut = (stride == 1 and in_channels == out_channels)

        # alias Swish
        activation_layer = SiLU
        
        # In EfficientNetV2, there is no expansion=1 in MBConv, so conv_pw must exist.
        assert expand_ratio != 1
        
        #expand width
        expanded_channels = in_channels * expand_ratio
                 
        
        self.expand_conv = ConvBNAct(in_channels      = in_channels,
                                     out_channels     = expanded_channels,
                                     kernel_size      = 1,
                                     norm_layer       = norm_layer,
                                     activation_layer = activation_layer)
      
        
        self.dwconv = ConvBNAct(in_channels      = expanded_channels,
                                out_channels     = expanded_channels,
                                kernel_size      = kernel_size,
                                stride           = stride,
                                groups           = expanded_channels,
                                norm_layer       = norm_layer,
                                activation_layer = activation_layer)

        self.SE = SqueezeExcite(input_channels   = in_channels,
                                expand_channels  = expanded_channels,
                                se_ratio         = se_ratio) if se_ratio > 0 else nn.Identity()
        
        
        self.project_conv = ConvBNAct(in_channels       = expanded_channels,
                                      out_channels      = out_channels,
                                      kernel_size       = 1,
                                      norm_layer        = norm_layer,
                                      activation_layer  = nn.Identity)  # There is no activation function, so use Identity (empty layer)


        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

            
    def forward(self, x: Tensor) -> Tensor:
        result = self.expand_conv(x)
        result = self.dwconv(result)
        result = self.SE(result)
        result = self.project_conv(result)

        #shortcut used condition.(stride == 1, in-channels == out-channels)
        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result  
    
    
class FusedMBConv(nn.Module):
    def __init__(self,
                 kernel_size: int,
                 in_channels: int,
                 out_channels: int,
                 expand_ratio: int,
                 stride: int,
                 se_ratio: float,
                 drop_rate: float,
                 norm_layer: Callable[..., nn.Module]):
        super(FusedMBConv, self).__init__()

        
        assert stride in [1, 2]
        assert se_ratio == 0

        self.out_channels = out_channels        
        self.drop_rate = drop_rate
        
        # shortcut used condition.(stride == 1, in-channels == out-channels)
        self.has_shortcut = (stride == 1 and in_channels == out_channels)

        # alias Swish
        activation_layer = SiLU  
        
        #expand width
        expanded_channels = in_channels * expand_ratio
        
        # Expand-Conv only exists when the expansion ratio !=1.
        self.has_expansion = (expand_ratio != 1)
        
             
        if self.has_expansion:           
            self.expand_conv = ConvBNAct(in_channels      = in_channels,
                                         out_channels     = expanded_channels,
                                         kernel_size      = kernel_size,
                                         stride           = stride,
                                         norm_layer       = norm_layer,
                                         activation_layer = activation_layer)
            
            self.project_conv = ConvBNAct(in_channels      = expanded_channels,
                                          out_channels     = out_channels,
                                          kernel_size      = 1,
                                          norm_layer       = norm_layer,
                                          activation_layer = nn.Identity)  # There is no activation function, so use Identity (empty layer)
            
       
        else:        
            self.project_conv = ConvBNAct(in_channels      = in_channels,
                                          out_channels     = out_channels,
                                          kernel_size      = kernel_size,
                                          stride           = stride,
                                          norm_layer       = norm_layer,
                                          activation_layer = activation_layer)  # There is activation function.

        
        # Use the dropout layer only when using the shortcut.
        if self.has_shortcut and drop_rate > 0:
            self.dropout = DropPath(drop_rate)

            
    def forward(self, x: Tensor) -> Tensor:
        if self.has_expansion:
            result = self.expand_conv(x)
            result = self.project_conv(result)
        else:
            result = self.project_conv(x)

        # shortcut used condition.(stride == 1, in-channels == out-channels)
        if self.has_shortcut:
            if self.drop_rate > 0:
                result = self.dropout(result)
            result += x

        return result
    
    
class EfficientNetV2(nn.Module):
    def __init__(self,
                 model_cnf: list,
                 num_classes: int = 1000,
                 num_features: int = 1280,
                 dropout_rate: float = 0.2,
                 drop_connect_rate: float = 0.2):
        super(EfficientNetV2, self).__init__()

        # The len of each layer of para must be 8.
        for cnf in model_cnf:
            assert len(cnf) == 8

        # Bind BatchNorm param, eps = 10^-3, momentum = 0.1
        norm_layer = partial(nn.BatchNorm2d, eps = 1e-3, momentum=0.1)
        
        
        self.stem = ConvBNAct(in_channels = 3,
                              out_channels = model_cnf[0][4],
                              kernel_size = 3,
                              stride = 2,
                              norm_layer = norm_layer)  # Default Act is SiLu
        
        
        total_blocks = sum([i[0] for i in model_cnf])
        block_id = 0
        blocks = []       
        for cnf in model_cnf:
            repeats = cnf[0]
            Operator_FMBConv = FusedMBConv if cnf[-2] == 0 else MBConv
            for i in range(repeats):
                blocks.append(Operator_FMBConv( kernel_size    = cnf[1],
                                                in_channels = cnf[4] if i == 0 else cnf[5],
                                                out_channels   = cnf[5],
                                                expand_ratio   = cnf[3],
                                                stride         = cnf[2] if i == 0 else 1,
                                                se_ratio       = cnf[-1],
                                                drop_rate      = drop_connect_rate * block_id / total_blocks,
                                                norm_layer     = norm_layer))
                block_id += 1                
        self.blocks = nn.Sequential(*blocks)
        
        
        head = OrderedDict()

        head.update({"project_conv": ConvBNAct(in_channels  = model_cnf[-1][-3],
                                               out_channels = num_features,
                                               kernel_size  = 1,
                                               norm_layer   = norm_layer)})  # Default Act is SiLu

        head.update({"avgpool": nn.AdaptiveAvgPool2d(1)})
        
        head.update({"flatten": nn.Flatten()})

        if dropout_rate > 0:
            head.update({"dropout": nn.Dropout(p = dropout_rate, inplace=True)})
            
        head.update({"classifier": nn.Linear(num_features, num_classes)})

        self.head = nn.Sequential(head)
        
        
        for m in self.modules():
            
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode = "fan_out")
                if m.bias is not None:
                    nn.init.zeros_(m.bias)
                    
            elif isinstance(m, nn.BatchNorm2d):
                nn.init.ones_(m.weight)
                nn.init.zeros_(m.bias)
                
            elif isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01)
                nn.init.zeros_(m.bias)

                
    def forward(self, x: Tensor) -> Tensor:
        x = self.stem(x)
        x = self.blocks(x)
        x = self.head(x)

        return x
    

def efficientnetv2_s(num_classes: int = 1000):

    
    model_config = [[2, 3, 1, 1, 24, 24, 0, 0],
                    [4, 3, 2, 4, 24, 48, 0, 0],
                    [4, 3, 2, 4, 48, 64, 0, 0],
                    [6, 3, 2, 4, 64, 128, 1, 0.25],
                    [9, 3, 1, 6, 128, 160, 1, 0.25],
                    [15, 3, 2, 6, 160, 256, 1, 0.25]]

    model = EfficientNetV2(model_cnf    = model_config,
                           num_classes  = num_classes,
                           dropout_rate = 0.2)
    return model

def efficientnetv2_m(num_classes: int = 1000):
    
    model_config = [[3, 3, 1, 1, 24, 24, 0, 0],
                    [5, 3, 2, 4, 24, 48, 0, 0],
                    [5, 3, 2, 4, 48, 80, 0, 0],
                    [7, 3, 2, 4, 80, 160, 1, 0.25],
                    [14, 3, 1, 6, 160, 176, 1, 0.25],
                    [18, 3, 2, 6, 176, 304, 1, 0.25],
                    [5, 3, 1, 6, 304, 512, 1, 0.25]]

    model = EfficientNetV2(model_cnf    = model_config,
                           num_classes  = num_classes,
                           dropout_rate = 0.3)
    return model

    
def efficientnetv2_l(num_classes: int = 1000):
    
    model_config = [[4, 3, 1, 1, 32, 32, 0, 0],
                    [7, 3, 2, 4, 32, 64, 0, 0],
                    [7, 3, 2, 4, 64, 96, 0, 0],
                    [10, 3, 2, 4, 96, 192, 1, 0.25],
                    [19, 3, 1, 6, 192, 224, 1, 0.25],
                    [25, 3, 2, 6, 224, 384, 1, 0.25],
                    [7, 3, 1, 6, 384, 640, 1, 0.25]]

    model = EfficientNetV2(model_cnf     = model_config,
                           num_classes   = num_classes,
                           dropout_rat   = 0.4)
    return model
    
    
    
    
    
    
    
    
