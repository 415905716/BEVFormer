# Copyright (c) OpenMMLab. All rights reserved.
from .alexnet import AlexNet
# yapf: disable
from .bricks import (ACTIVATION_LAYERS, CONV_LAYERS, NORM_LAYERS,
                     PADDING_LAYERS, PLUGIN_LAYERS, UPSAMPLE_LAYERS,
                     ContextBlock, Conv2d, Conv3d, ConvAWS2d, ConvModule,
                     ConvTranspose2d, ConvTranspose3d, ConvWS2d,
                     DepthwiseSeparableConvModule, GeneralizedAttention,
                     HSigmoid, HSwish, Linear, MaxPool2d, MaxPool3d,
                     NonLocal1d, NonLocal2d, NonLocal3d, Scale, Swish,Linear_Q,Linear_Q_tensorboard,
                     build_activation_layer, build_conv_layer,
                     build_norm_layer, build_padding_layer, build_plugin_layer,
                     build_upsample_layer, conv_ws_2d, is_norm)
from .builder import MODELS, build_model_from_cfg
# yapf: enable
from .resnet import ResNet, make_res_layer
from .utils import (INITIALIZERS, Caffe2XavierInit, ConstantInit, KaimingInit,
                    NormalInit, PretrainedInit, TruncNormalInit, UniformInit,
                    XavierInit, bias_init_with_prob, caffe2_xavier_init,
                    constant_init, fuse_conv_bn, get_model_complexity_info,
                    initialize, kaiming_init, normal_init, trunc_normal_init,
                    uniform_init, xavier_init)
from .vgg import VGG, make_vgg_layer

__all__ = [
    'AlexNet', 'VGG', 'make_vgg_layer', 'ResNet', 'make_res_layer',
    'constant_init', 'xavier_init', 'normal_init', 'trunc_normal_init',
    'uniform_init', 'kaiming_init', 'caffe2_xavier_init',
    'bias_init_with_prob', 'ConvModule', 'build_activation_layer',
    'build_conv_layer', 'build_norm_layer', 'build_padding_layer',
    'build_upsample_layer', 'build_plugin_layer', 'is_norm', 'NonLocal1d',
    'NonLocal2d', 'NonLocal3d', 'ContextBlock', 'HSigmoid', 'Swish', 'HSwish',
    'GeneralizedAttention', 'ACTIVATION_LAYERS', 'CONV_LAYERS', 'NORM_LAYERS',
    'PADDING_LAYERS', 'UPSAMPLE_LAYERS', 'PLUGIN_LAYERS', 'Scale',
    'get_model_complexity_info', 'conv_ws_2d', 'ConvAWS2d', 'ConvWS2d',
    'fuse_conv_bn', 'DepthwiseSeparableConvModule', 'Linear', 'Conv2d',
    'ConvTranspose2d', 'MaxPool2d', 'ConvTranspose3d', 'MaxPool3d', 'Conv3d',
    'initialize', 'INITIALIZERS', 'ConstantInit', 'XavierInit', 'NormalInit',
    'TruncNormalInit', 'UniformInit', 'KaimingInit', 'PretrainedInit',
    'Caffe2XavierInit', 'MODELS', 'build_model_from_cfg','Linear_Q','Linear_Q_tensorboard'
]

















# Copyright (c) OpenMMLab. All rights reserved.
from .activation import build_activation_layer
from .context_block import ContextBlock
from .conv import build_conv_layer, Linear_Q_tensorboard, Linear_Q
from .conv2d_adaptive_padding import Conv2dAdaptivePadding
from .conv_module import ConvModule
from .conv_ws import ConvAWS2d, ConvWS2d, conv_ws_2d
from .depthwise_separable_conv_module import DepthwiseSeparableConvModule
from .drop import Dropout, DropPath
from .generalized_attention import GeneralizedAttention
from .hsigmoid import HSigmoid
from .hswish import HSwish
from .non_local import NonLocal1d, NonLocal2d, NonLocal3d
from .norm import build_norm_layer, is_norm
from .padding import build_padding_layer
from .plugin import build_plugin_layer
from .registry import (ACTIVATION_LAYERS, CONV_LAYERS, NORM_LAYERS,
                       PADDING_LAYERS, PLUGIN_LAYERS, UPSAMPLE_LAYERS)
from .scale import Scale
from .swish import Swish
from .upsample import build_upsample_layer
from .wrappers import (Conv2d, Conv3d, ConvTranspose2d, ConvTranspose3d,
                       Linear, MaxPool2d, MaxPool3d)

__all__ = [
    'ConvModule', 'build_activation_layer', 'build_conv_layer',
    'build_norm_layer', 'build_padding_layer', 'build_upsample_layer',
    'build_plugin_layer', 'is_norm', 'HSigmoid', 'HSwish', 'NonLocal1d',
    'NonLocal2d', 'NonLocal3d', 'ContextBlock', 'GeneralizedAttention',
    'ACTIVATION_LAYERS', 'CONV_LAYERS', 'NORM_LAYERS', 'PADDING_LAYERS',
    'UPSAMPLE_LAYERS', 'PLUGIN_LAYERS', 'Scale', 'ConvAWS2d', 'ConvWS2d',
    'conv_ws_2d', 'DepthwiseSeparableConvModule', 'Swish', 'Linear',
    'Conv2dAdaptivePadding', 'Conv2d', 'ConvTranspose2d', 'MaxPool2d',
    'ConvTranspose3d', 'MaxPool3d', 'Conv3d', 'Dropout', 'DropPath','Linear_Q_tensorboard', 'Linear_Q'
]





















# Copyright (c) OpenMMLab. All rights reserved.
from tkinter.messagebox import QUESTION
from torch import nn

from .registry import CONV_LAYERS

import torch
import torch.nn.functional as F
import math
from torch.autograd import Function

import sys
import os
sys.path.append(r"/home/zhangyifan/BEVFormer/HAWQ")

from HAWQ.utils.quantization_utils.quant_utils import AsymmetricQuantFunction, SymmetricQuantFunction, get_percentile_min_max, symmetric_linear_quantization_params
from HAWQ.utils.quantization_utils.quant_modules import QuantAct

class Linear_Q(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 weight_bit=6,
                 bias_bit=32,
                 full_precision_flag=True,
                 quant_mode='symmetric',
                 per_channel=False,
                 fix_flag=False,
                 weight_percentile=99.95,
                 
                 quant_act=True,
                 activation_bit = 6,
                 act_percentile=0
                 ):
        super(Linear_Q, self).__init__(in_features, out_features, bias)
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode
        self.counter = 0
        
        self.quant_act = quant_act
        self.activation_bit = activation_bit
        self.act_percentile = act_percentile
        self.QuantAct = QuantAct(activation_bit=self.activation_bit, act_percentile=self.act_percentile)

    def fix(self):
        self.fix_flag = True

    def unfix(self):
        self.fix_flag = False

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        # writer.add_histogram("x_before_linear", x, )
        # from tensorboardX import SummaryWriter
        # writer = SummaryWriter('tensorboard/act_quant/')
        
        # writer.add_histogram('x_before_quant', x, 0) 
        if type(x) is tuple:
            x = x[0]
            
        if self.quant_act:
            x = self.QuantAct(x)
            
        if type(x) is tuple:
            x = x[0]
        
         
        # writer.add_histogram('x_after_quant', x, 0) 
        # print("linear_Q")
        # import pdb; pdb.set_trace()
        # import pdb; pdb.set_trace()
        
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        w = self.weight
        w_transform = w.data.detach()
        # calculate the quantization range of weights and bias
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        # perform the quantization
        if not self.full_precision_flag:
            if self.quant_mode == 'symmetric':
                self.fc_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max,
                                                                              self.per_channel)
                self.weight_integer = self.weight_function(self.weight, self.weight_bit, self.fc_scaling_factor)

                # bias_scaling_factor = self.fc_scaling_factor.view(1, -1) * prev_act_scaling_factor.view(1, -1)
                bias_scaling_factor = self.fc_scaling_factor.view(1, -1)
                self.bias_integer = self.weight_function(self.bias, self.bias_bit, bias_scaling_factor)
            else:
                raise Exception('For weight, we only support symmetric quantization.')
        else:
            w = self.weight
            b = self.bias
            return F.linear(x, weight=w, bias=b)

        correct_output_scale = bias_scaling_factor[0].view(1, -1)
        
        return F.linear(x, weight=self.weight_integer, bias=self.bias_integer) * correct_output_scale

class Linear_Q_tensorboard(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 weight_bit=6,
                 bias_bit=32,
                 full_precision_flag=False,
                 quant_mode='symmetric',
                 per_channel=False,
                 fix_flag=False,
                 weight_percentile=99.95,
                 
                 quant_act=True,
                 activation_bit = 6,
                 act_percentile=0
                 ):
        super(Linear_Q_tensorboard, self).__init__(in_features, out_features, bias)
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode
        self.counter = 0
        
        self.quant_act = quant_act
        self.activation_bit = activation_bit
        self.act_percentile = act_percentile
        self.QuantAct = QuantAct(activation_bit=self.activation_bit, act_percentile=self.act_percentile)

    def fix(self):
        self.fix_flag = True

    def unfix(self):
        self.fix_flag = False

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        from tensorboardX import SummaryWriter
        import matplotlib.pyplot as plt
        
        writer = SummaryWriter('tensorboard/tsa_ap0.05/')
        # writer.add_histogram('x_before_quant', x, 0.05) 
        
        
        x_before_quant = x
        
        
        if type(x) is tuple:
            x = x[0]
            
        if self.quant_act:
            x = self.QuantAct(x)
            
        if type(x) is tuple:
            x = x[0]
         
        # writer.add_histogram('x_after_quant', x, 0)  
        
        
        print("linear_Q_tensorboard")
        import pdb; pdb.set_trace()

        
        
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        w = self.weight
        w_transform = w.data.detach()
        # calculate the quantization range of weights and bias
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        # perform the quantization
        if not self.full_precision_flag:
            if self.quant_mode == 'symmetric':
                self.fc_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max,
                                                                              self.per_channel)
                self.weight_integer = self.weight_function(self.weight, self.weight_bit, self.fc_scaling_factor)

                # bias_scaling_factor = self.fc_scaling_factor.view(1, -1) * prev_act_scaling_factor.view(1, -1)
                bias_scaling_factor = self.fc_scaling_factor.view(1, -1)
                self.bias_integer = self.weight_function(self.bias, self.bias_bit, bias_scaling_factor)
            else:
                raise Exception('For weight, we only support symmetric quantization.')
        else:
            w = self.weight
            b = self.bias
            return F.linear(x, weight=w, bias=b)

        correct_output_scale = bias_scaling_factor[0].view(1, -1)
        
        return F.linear(x, weight=self.weight_integer, bias=self.bias_integer) * correct_output_scale
    
class Quan_Conv2d(nn.Conv2d):
    def __init__(self, in_channels, out_channels, kernel_size, stride=1, padding=0, dilation=1, groups=1, bias=True,
                 weight_bit=6,
                 bias_bit=None,
                 full_precision_flag=False,
                 quant_mode="symmetric",
                 per_channel=False,
                 fix_flag=False,
                 weight_percentile=99.95,
                 
                 quant_act=True,
                 activation_bit = 6,
                 act_percentile=0):
        super(Quan_Conv2d, self).__init__( in_channels, out_channels, kernel_size, stride, padding, dilation,
            groups, bias)
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        
        self.quant_act = quant_act
        self.activation_bit = activation_bit
        self.act_percentile = act_percentile
        self.QuantAct = QuantAct(activation_bit=self.activation_bit,act_percentile=self.act_percentile)
        
        
    def forward(self, x):
        
        if self.quant_act:
            # print("already quant_act")
            x = self.QuantAct(x)
            
        if type(x) is tuple:
            x = x[0]
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        w = self.weight

        # calculate quantization range
        if self.per_channel:
            w_transform = w.data.contiguous().view(self.out_channels, -1)

            if self.weight_percentile == 0:
                w_min = w_transform.min(dim=1).values
                w_max = w_transform.max(dim=1).values
            else:
                lower_percentile = 100 - self.weight_percentile
                upper_percentile = self.weight_percentile
                input_length = w_transform.shape[1]

                lower_index = math.ceil(input_length * lower_percentile * 0.01)
                upper_index = math.ceil(input_length * upper_percentile * 0.01)

                w_min = torch.kthvalue(w_transform, k=lower_index, dim=1).values
                w_max = torch.kthvalue(w_transform, k=upper_index, dim=1).values
        elif not self.per_channel:
            if self.weight_percentile == 0:
                w_min = w.data.min()
                w_max = w.data.max()
            else:
                w_min, w_max = get_percentile_min_max(w.view(-1), 100 - self.weight_percentile,
                                                      self.weight_percentile, output_tensor=True)
        # perform quantization
        if self.quant_mode == 'symmetric':
            self.conv_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max,
                                                                            self.per_channel)
            self.weight_integer = self.weight_function(self.weight, self.weight_bit, self.conv_scaling_factor)
            bias_scaling_factor = self.conv_scaling_factor.view(1, -1)
            if self.quantize_bias and (self.bias is not None):
                self.bias_integer = self.weight_function(self.bias, self.bias_bit, bias_scaling_factor)
            else:
                self.bias_integer = None
        else:
            raise Exception('For weight, we only support symmetric quantization.')

        correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)
        
        if self.bias is None:
            output = F.conv2d(x, self.weight_integer, self.bias, self.stride, self.padding, self.dilation, self.groups)*correct_output_scale
            # output = F.conv2d(x, self.weight_integer, torch.zeros_like(self.bias), self.stride, self.padding, self.dilation, self.groups)*correct_output_scale
        else:
            output = F.conv2d(x, self.weight_integer, self.bias_integer, self.stride, self.padding, self.dilation, self.groups)* correct_output_scale
        return output
        # if self.bias is None:
        #     return (F.conv2d(x_int, self.weight_integer, torch.zeros_like(bias_scaling_factor.view(-1)),
        #                      self.conv.stride, self.conv.padding, self.conv.dilation, self.conv.groups)
        #             * correct_output_scale, self.conv_scaling_factor)
        # else:
        #     return (F.conv2d(x_int, self.weight_integer, self.bias_integer, 
        #                      self.conv.stride, self.conv.padding,self.conv.dilation, self.conv.groups) 
        #             * correct_output_scale, self.conv_scaling_factor)
        

CONV_LAYERS.register_module('Conv1d', module=nn.Conv1d)
CONV_LAYERS.register_module('Conv2d', module=nn.Conv2d)
CONV_LAYERS.register_module('Conv3d', module=nn.Conv3d)
CONV_LAYERS.register_module('Conv', module=Quan_Conv2d)


def build_conv_layer(cfg, *args, **kwargs):
    """Build convolution layer.

    Args:
        cfg (None or dict): The conv layer config, which should contain:
            - type (str): Layer type.
            - layer args: Args needed to instantiate an conv layer.
        args (argument list): Arguments passed to the `__init__`
            method of the corresponding conv layer.
        kwargs (keyword arguments): Keyword arguments passed to the `__init__`
            method of the corresponding conv layer.

    Returns:
        nn.Module: Created conv layer.
    """
    if cfg is None:
        cfg_ = dict(type='Conv2d')
    else:
        if not isinstance(cfg, dict):
            raise TypeError('cfg must be a dict')
        if 'type' not in cfg:
            raise KeyError('the cfg dict must contain the key "type"')
        cfg_ = cfg.copy()

    layer_type = cfg_.pop('type')
    if layer_type not in CONV_LAYERS:
        raise KeyError(f'Unrecognized norm type {layer_type}')
    else:
        conv_layer = CONV_LAYERS.get(layer_type)

    layer = conv_layer(*args, **kwargs, **cfg_)

    return layer


    
    
    






















# Copyright (c) OpenMMLab. All rights reserved.
r"""Modified from https://github.com/facebookresearch/detectron2/blob/master/detectron2/layers/wrappers.py  # noqa: E501

Wrap some nn modules to support empty tensor input. Currently, these wrappers
are mainly used in mask heads like fcn_mask_head and maskiou_heads since mask
heads are trained on only positive RoIs.
"""
import math

import torch
import torch.nn as nn
from torch.nn.modules.utils import _pair, _triple

from .registry import CONV_LAYERS, UPSAMPLE_LAYERS

if torch.__version__ == 'parrots':
    TORCH_VERSION = torch.__version__
else:
    # torch.__version__ could be 1.3.1+cu92, we only need the first two
    # for comparison
    TORCH_VERSION = tuple(int(x) for x in torch.__version__.split('.')[:2])


def obsolete_torch_version(torch_version, version_threshold):
    return torch_version == 'parrots' or torch_version <= version_threshold


class NewEmptyTensorOp(torch.autograd.Function):

    @staticmethod
    def forward(ctx, x, new_shape):
        ctx.shape = x.shape
        return x.new_empty(new_shape)

    @staticmethod
    def backward(ctx, grad):
        shape = ctx.shape
        return NewEmptyTensorOp.apply(grad, shape), None


@CONV_LAYERS.register_module('Conv', force=True)
class Conv2d(nn.Conv2d):

    def forward(self, x):
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 4)):
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d in zip(x.shape[-2:], self.kernel_size,
                                     self.padding, self.stride, self.dilation):
                o = (i + 2 * p - (d * (k - 1) + 1)) // s + 1
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)


@CONV_LAYERS.register_module('Conv3d', force=True)
class Conv3d(nn.Conv3d):

    def forward(self, x):
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 4)):
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d in zip(x.shape[-3:], self.kernel_size,
                                     self.padding, self.stride, self.dilation):
                o = (i + 2 * p - (d * (k - 1) + 1)) // s + 1
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)


@CONV_LAYERS.register_module()
@CONV_LAYERS.register_module('deconv')
@UPSAMPLE_LAYERS.register_module('deconv', force=True)
class ConvTranspose2d(nn.ConvTranspose2d):

    def forward(self, x):
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 4)):
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d, op in zip(x.shape[-2:], self.kernel_size,
                                         self.padding, self.stride,
                                         self.dilation, self.output_padding):
                out_shape.append((i - 1) * s - 2 * p + (d * (k - 1) + 1) + op)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)


@CONV_LAYERS.register_module()
@CONV_LAYERS.register_module('deconv3d')
@UPSAMPLE_LAYERS.register_module('deconv3d', force=True)
class ConvTranspose3d(nn.ConvTranspose3d):

    def forward(self, x):
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 4)):
            out_shape = [x.shape[0], self.out_channels]
            for i, k, p, s, d, op in zip(x.shape[-3:], self.kernel_size,
                                         self.padding, self.stride,
                                         self.dilation, self.output_padding):
                out_shape.append((i - 1) * s - 2 * p + (d * (k - 1) + 1) + op)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            if self.training:
                # produce dummy gradient to avoid DDP warning.
                dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
                return empty + dummy
            else:
                return empty

        return super().forward(x)


class MaxPool2d(nn.MaxPool2d):

    def forward(self, x):
        # PyTorch 1.9 does not support empty tensor inference yet
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 9)):
            out_shape = list(x.shape[:2])
            for i, k, p, s, d in zip(x.shape[-2:], _pair(self.kernel_size),
                                     _pair(self.padding), _pair(self.stride),
                                     _pair(self.dilation)):
                o = (i + 2 * p - (d * (k - 1) + 1)) / s + 1
                o = math.ceil(o) if self.ceil_mode else math.floor(o)
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            return empty

        return super().forward(x)


class MaxPool3d(nn.MaxPool3d):

    def forward(self, x):
        # PyTorch 1.9 does not support empty tensor inference yet
        if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 9)):
            out_shape = list(x.shape[:2])
            for i, k, p, s, d in zip(x.shape[-3:], _triple(self.kernel_size),
                                     _triple(self.padding),
                                     _triple(self.stride),
                                     _triple(self.dilation)):
                o = (i + 2 * p - (d * (k - 1) + 1)) / s + 1
                o = math.ceil(o) if self.ceil_mode else math.floor(o)
                out_shape.append(o)
            empty = NewEmptyTensorOp.apply(x, out_shape)
            return empty

        return super().forward(x)


# class Linear(torch.nn.Linear):

#     def forward(self, x):
#         # empty tensor forward of Linear layer is supported in Pytorch 1.6
#         if x.numel() == 0 and obsolete_torch_version(TORCH_VERSION, (1, 5)):
#             out_shape = [x.shape[0], self.out_features]
#             empty = NewEmptyTensorOp.apply(x, out_shape)
#             if self.training:
#                 # produce dummy gradient to avoid DDP warning.
#                 dummy = sum(x.view(-1)[0] for x in self.parameters()) * 0.0
#                 return empty + dummy
#             else:
#                 return empty

#         return super().forward(x)

import torch.nn.functional as F
from torch import nn
from HAWQ.utils.quantization_utils.quant_utils import AsymmetricQuantFunction, SymmetricQuantFunction, get_percentile_min_max, symmetric_linear_quantization_params
from HAWQ.utils.quantization_utils.quant_modules import QuantAct

class Linear(nn.Linear):
    def __init__(self, in_features, out_features, bias=True,
                 weight_bit=6,
                 bias_bit=32,
                 full_precision_flag=True,
                 quant_mode='symmetric',
                 per_channel=False,
                 fix_flag=False,
                 weight_percentile=99.95,
                 
                 quant_act=True,
                 activation_bit = 6,
                 act_percentile=0
                 ):
        super(Linear, self).__init__(in_features, out_features, bias)
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.quant_mode = quant_mode
        self.counter = 0
        
        self.quant_act = quant_act
        self.activation_bit = activation_bit
        self.act_percentile = act_percentile
        self.QuantAct = QuantAct(activation_bit=self.activation_bit, act_percentile=self.act_percentile)

    def fix(self):
        self.fix_flag = True

    def unfix(self):
        self.fix_flag = False

    def forward(self, x):
        """
        using quantized weights to forward activation x
        """
        # writer.add_histogram("x_before_linear", x, )
        if type(x) is tuple:
            x = x[0]
            
        if self.quant_act:
            x = self.QuantAct(x)
            
        if type(x) is tuple:
            x = x[0]
        
        
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        w = self.weight
        w_transform = w.data.detach()
        # calculate the quantization range of weights and bias
        if self.per_channel:
            w_min, _ = torch.min(w_transform, dim=1, out=None)
            w_max, _ = torch.max(w_transform, dim=1, out=None)
        else:
            w_min = w_transform.min().expand(1)
            w_max = w_transform.max().expand(1)

        # perform the quantization
        if not self.full_precision_flag:
            if self.quant_mode == 'symmetric':
                self.fc_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max,
                                                                              self.per_channel)
                self.weight_integer = self.weight_function(self.weight, self.weight_bit, self.fc_scaling_factor)

                # bias_scaling_factor = self.fc_scaling_factor.view(1, -1) * prev_act_scaling_factor.view(1, -1)
                bias_scaling_factor = self.fc_scaling_factor.view(1, -1)
                self.bias_integer = self.weight_function(self.bias, self.bias_bit, bias_scaling_factor)
            else:
                raise Exception('For weight, we only support symmetric quantization.')
        else:
            w = self.weight
            b = self.bias
            return F.linear(x, weight=w, bias=b)

        correct_output_scale = bias_scaling_factor[0].view(1, -1)
        
        return F.linear(x, weight=self.weight_integer, bias=self.bias_integer) * correct_output_scale




























# Copyright (c) OpenMMLab. All rights reserved.
from .base import Base3DDetector
from .centerpoint import CenterPoint
from .dynamic_voxelnet import DynamicVoxelNet
from .fcos_mono3d import FCOSMono3D
from .groupfree3dnet import GroupFree3DNet
from .h3dnet import H3DNet
from .imvotenet import ImVoteNet
from .imvoxelnet import ImVoxelNet
from .mvx_faster_rcnn import DynamicMVXFasterRCNN, MVXFasterRCNN
from .mvx_two_stage import MVXTwoStageDetector
from .parta2 import PartA2
from .single_stage_mono3d import SingleStageMono3DDetector
from .ssd3dnet import SSD3DNet
from .votenet import VoteNet
from .voxelnet import VoxelNet
from .bevformer1 import BEVFormer1

__all__ = [
    'Base3DDetector', 'VoxelNet', 'DynamicVoxelNet', 'MVXTwoStageDetector',
    'DynamicMVXFasterRCNN', 'MVXFasterRCNN', 'PartA2', 'VoteNet', 'H3DNet',
    'CenterPoint', 'SSD3DNet', 'ImVoteNet', 'SingleStageMono3DDetector',
    'FCOSMono3D', 'ImVoxelNet', 'GroupFree3DNet', 'BEVFormer1'
]






































# ---------------------------------------------
# Copyright (c) OpenMMLab. All rights reserved.
# ---------------------------------------------
#  Modified by Zhiqi Li
# ---------------------------------------------
import torch
from mmcv.runner import force_fp32, auto_fp16
from mmdet.models import DETECTORS
from mmdet3d.core import bbox3d2result
from mmdet3d.models.detectors.mvx_two_stage import MVXTwoStageDetector
import copy
import numpy as np
from PIL import Image

class GridMask(torch.nn.Module):
    def __init__(self, use_h, use_w, rotate = 1, offset=False, ratio = 0.5, mode=0, prob = 1.):
        super(GridMask, self).__init__()
        self.use_h = use_h
        self.use_w = use_w
        self.rotate = rotate
        self.offset = offset
        self.ratio = ratio
        self.mode = mode
        self.st_prob = prob
        self.prob = prob
        self.fp16_enable = False
    def set_prob(self, epoch, max_epoch):
        self.prob = self.st_prob * epoch / max_epoch #+ 1.#0.5
    @auto_fp16()
    def forward(self, x):
        if np.random.rand() > self.prob or not self.training:
            return x
        n,c,h,w = x.size()
        x = x.view(-1,h,w)
        hh = int(1.5*h)
        ww = int(1.5*w)
        d = np.random.randint(2, h)
        self.l = min(max(int(d*self.ratio+0.5),1),d-1)
        mask = np.ones((hh, ww), np.float32)
        st_h = np.random.randint(d)
        st_w = np.random.randint(d)
        if self.use_h:
            for i in range(hh//d):
                s = d*i + st_h
                t = min(s+self.l, hh)
                mask[s:t,:] *= 0
        if self.use_w:
            for i in range(ww//d):
                s = d*i + st_w
                t = min(s+self.l, ww)
                mask[:,s:t] *= 0
       
        r = np.random.randint(self.rotate)
        mask = Image.fromarray(np.uint8(mask))
        mask = mask.rotate(r)
        mask = np.asarray(mask)
        mask = mask[(hh-h)//2:(hh-h)//2+h, (ww-w)//2:(ww-w)//2+w]

        mask = torch.from_numpy(mask).to(x.dtype).cuda()
        if self.mode == 1:
            mask = 1-mask
        mask = mask.expand_as(x)
        if self.offset:
            offset = torch.from_numpy(2 * (np.random.rand(h,w) - 0.5)).to(x.dtype).cuda()
            x = x * mask + offset * (1 - mask)
        else:
            x = x * mask 
        
        return x.view(n,c,h,w)


@DETECTORS.register_module()
class BEVFormer1(MVXTwoStageDetector):
    """BEVFormer.
    Args:
        video_test_mode (bool): Decide whether to use temporal information during inference.
    """

    def __init__(self,
                 use_grid_mask=False,
                 pts_voxel_layer=None,
                 pts_voxel_encoder=None,
                 pts_middle_encoder=None,
                 pts_fusion_layer=None,
                 img_backbone=None,
                 pts_backbone=None,
                 img_neck=None,
                 pts_neck=None,
                 pts_bbox_head=None,
                 img_roi_head=None,
                 img_rpn_head=None,
                 train_cfg=None,
                 test_cfg=None,
                 pretrained=None,
                 video_test_mode=False,
                 ):

        super(BEVFormer1,
              self).__init__(pts_voxel_layer, pts_voxel_encoder,
                             pts_middle_encoder, pts_fusion_layer,
                             img_backbone, pts_backbone, img_neck, pts_neck,
                             pts_bbox_head, img_roi_head, img_rpn_head,
                             train_cfg, test_cfg, pretrained)
        self.grid_mask = GridMask(
            True, True, rotate=1, offset=False, ratio=0.5, mode=1, prob=0.7)
        self.use_grid_mask = use_grid_mask
        self.fp16_enabled = False

        # temporal
        self.video_test_mode = video_test_mode
        self.prev_frame_info = {
            'prev_bev': None,
            'scene_token': None,
            'prev_pos': 0,
            'prev_angle': 0,
        }


    def extract_img_feat(self, img, img_metas, len_queue=None):
        """Extract features of images."""
        B = img.size(0)
        if img is not None:
            
            # input_shape = img.shape[-2:]
            # # update real input shape of each single img
            # for img_meta in img_metas:
            #     img_meta.update(input_shape=input_shape)

            if img.dim() == 5 and img.size(0) == 1:
                img.squeeze_()
            elif img.dim() == 5 and img.size(0) > 1:
                B, N, C, H, W = img.size()
                img = img.reshape(B * N, C, H, W)
            if self.use_grid_mask:
                img = self.grid_mask(img)

            img_feats = self.img_backbone(img)
            if isinstance(img_feats, dict):
                img_feats = list(img_feats.values())
        else:
            return None
        if self.with_img_neck:
            img_feats = self.img_neck(img_feats)

        img_feats_reshaped = []
        for img_feat in img_feats:
            BN, C, H, W = img_feat.size()
            if len_queue is not None:
                img_feats_reshaped.append(img_feat.view(int(B/len_queue), len_queue, int(BN / B), C, H, W))
            else:
                img_feats_reshaped.append(img_feat.view(B, int(BN / B), C, H, W))
        return img_feats_reshaped

    @auto_fp16(apply_to=('img'))
    def extract_feat(self, img, img_metas=None, len_queue=None):
        """Extract features from images and points."""

        img_feats = self.extract_img_feat(img, img_metas, len_queue=len_queue)
        
        return img_feats


    def forward_pts_train(self,
                          pts_feats,
                          gt_bboxes_3d,
                          gt_labels_3d,
                          img_metas,
                          gt_bboxes_ignore=None,
                          prev_bev=None):
        """Forward function'
        Args:
            pts_feats (list[torch.Tensor]): Features of point cloud branch
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`]): Ground truth
                boxes for each sample.
            gt_labels_3d (list[torch.Tensor]): Ground truth labels for
                boxes of each sampole
            img_metas (list[dict]): Meta information of samples.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                boxes to be ignored. Defaults to None.
            prev_bev (torch.Tensor, optional): BEV features of previous frame.
        Returns:
            dict: Losses of each branch.
        """

        outs = self.pts_bbox_head(
            pts_feats, img_metas, prev_bev)
        loss_inputs = [gt_bboxes_3d, gt_labels_3d, outs]
        losses = self.pts_bbox_head.loss(*loss_inputs, img_metas=img_metas)
        return losses

    def forward_dummy(self, img):
        dummy_metas = None
        return self.forward_test(img=img, img_metas=[[dummy_metas]])

    def forward(self, return_loss=True, **kwargs):
        """Calls either forward_train or forward_test depending on whether
        return_loss=True.
        Note this setting will change the expected inputs. When
        `return_loss=True`, img and img_metas are single-nested (i.e.
        torch.Tensor and list[dict]), and when `resturn_loss=False`, img and
        img_metas should be double nested (i.e.  list[torch.Tensor],
        list[list[dict]]), with the outer list indicating test time
        augmentations.
        """
        # import pdb; pdb.set_trace()
        # print("return_loss_student")
        # print(return_loss)
        # return_loss = return_loss
        if return_loss:
            return self.forward_train(**kwargs)
        else:
            return self.forward_test(**kwargs)
    
    def obtain_history_bev(self, imgs_queue, img_metas_list):
        """Obtain history BEV features iteratively. To save GPU memory, gradients are not calculated.
        """
        self.eval()

        with torch.no_grad():
            prev_bev = None
            bs, len_queue, num_cams, C, H, W = imgs_queue.shape
            imgs_queue = imgs_queue.reshape(bs*len_queue, num_cams, C, H, W)
            img_feats_list = self.extract_feat(img=imgs_queue, len_queue=len_queue)
            for i in range(len_queue):
                img_metas = [each[i] for each in img_metas_list]
                # img_feats = self.extract_feat(img=img, img_metas=img_metas)
                img_feats = [each_scale[:, i] for each_scale in img_feats_list]
                prev_bev = self.pts_bbox_head(
                    img_feats, img_metas, prev_bev, only_bev=True)
            self.train()
            return prev_bev

    @auto_fp16(apply_to=('img', 'points'))
    def forward_train(self,
                      points=None,
                      img_metas=None,
                      gt_bboxes_3d=None,
                      gt_labels_3d=None,
                      gt_labels=None,
                      gt_bboxes=None,
                      img=None,
                      proposals=None,
                      gt_bboxes_ignore=None,
                      img_depth=None,
                      img_mask=None,
                      ):
        """Forward training function.
        Args:
            points (list[torch.Tensor], optional): Points of each sample.
                Defaults to None.
            img_metas (list[dict], optional): Meta information of each sample.
                Defaults to None.
            gt_bboxes_3d (list[:obj:`BaseInstance3DBoxes`], optional):
                Ground truth 3D boxes. Defaults to None.
            gt_labels_3d (list[torch.Tensor], optional): Ground truth labels
                of 3D boxes. Defaults to None.
            gt_labels (list[torch.Tensor], optional): Ground truth labels
                of 2D boxes in images. Defaults to None.
            gt_bboxes (list[torch.Tensor], optional): Ground truth 2D boxes in
                images. Defaults to None.
            img (torch.Tensor optional): Images of each sample with shape
                (N, C, H, W). Defaults to None.
            proposals ([list[torch.Tensor], optional): Predicted proposals
                used for training Fast RCNN. Defaults to None.
            gt_bboxes_ignore (list[torch.Tensor], optional): Ground truth
                2D boxes in images to be ignored. Defaults to None.
        Returns:
            dict: Losses of different branches.
        """
        
        len_queue = img.size(1)
        prev_img = img[:, :-1, ...]
        img = img[:, -1, ...]

        prev_img_metas = copy.deepcopy(img_metas)
        prev_bev = self.obtain_history_bev(prev_img, prev_img_metas)

        img_metas = [each[len_queue-1] for each in img_metas]
        img_feats = self.extract_feat(img=img, img_metas=img_metas)
        losses = dict()
        losses_pts = self.forward_pts_train(img_feats, gt_bboxes_3d,
                                            gt_labels_3d, img_metas,
                                            gt_bboxes_ignore, prev_bev)

        losses.update(losses_pts)
        return losses

    def forward_test(self, img_metas, img=None, **kwargs):
        for var, name in [(img_metas, 'img_metas')]:
            if not isinstance(var, list):
                raise TypeError('{} must be a list, but got {}'.format(
                    name, type(var)))
        img = [img] if img is None else img

        if img_metas[0][0]['scene_token'] != self.prev_frame_info['scene_token']:
            # the first sample of each scene is truncated
            self.prev_frame_info['prev_bev'] = None
        # update idx
        self.prev_frame_info['scene_token'] = img_metas[0][0]['scene_token']

        # do not use temporal information
        if not self.video_test_mode:
            self.prev_frame_info['prev_bev'] = None

        # Get the delta of ego position and angle between two timestamps.
        tmp_pos = copy.deepcopy(img_metas[0][0]['can_bus'][:3])
        tmp_angle = copy.deepcopy(img_metas[0][0]['can_bus'][-1])
        if self.prev_frame_info['prev_bev'] is not None:
            img_metas[0][0]['can_bus'][:3] -= self.prev_frame_info['prev_pos']
            img_metas[0][0]['can_bus'][-1] -= self.prev_frame_info['prev_angle']
        else:
            img_metas[0][0]['can_bus'][-1] = 0
            img_metas[0][0]['can_bus'][:3] = 0

        new_prev_bev, bbox_results = self.simple_test(
            img_metas[0], img[0], prev_bev=self.prev_frame_info['prev_bev'], **kwargs)
        # During inference, we save the BEV features and ego motion of each timestamp.
        self.prev_frame_info['prev_pos'] = tmp_pos
        self.prev_frame_info['prev_angle'] = tmp_angle
        self.prev_frame_info['prev_bev'] = new_prev_bev
        return bbox_results

    def simple_test_pts(self, x, img_metas, prev_bev=None, rescale=False):
        """Test function"""
        outs = self.pts_bbox_head(x, img_metas, prev_bev=prev_bev)

        bbox_list = self.pts_bbox_head.get_bboxes(
            outs, img_metas, rescale=rescale)
        bbox_results = [
            bbox3d2result(bboxes, scores, labels)
            for bboxes, scores, labels in bbox_list
        ]
        return outs['bev_embed'], bbox_results

    def simple_test(self, img_metas, img=None, prev_bev=None, rescale=False):
        """Test function without augmentaiton."""
        img_feats = self.extract_feat(img=img, img_metas=img_metas)

        bbox_list = [dict() for i in range(len(img_metas))]
        new_prev_bev, bbox_pts = self.simple_test_pts(
            img_feats, img_metas, prev_bev, rescale=rescale)
        for result_dict, pts_bbox in zip(bbox_list, bbox_pts):
            result_dict['pts_bbox'] = pts_bbox
        return new_prev_bev, bbox_list
