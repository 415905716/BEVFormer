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
                 full_precision_flag=False,
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
        from tensorboardX import SummaryWriter
        writer = SummaryWriter('tensorboard/tsa/')
        writer.add_histogram('x_before_quant', x, 0) 
        
        if type(x) is tuple:
            x = x[0]
            
        if self.quant_act:
            x = self.QuantAct(x)
            
        if type(x) is tuple:
            x = x[0]
         
        writer.add_histogram('x_after_quant', x, 0)  
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
CONV_LAYERS.register_module('Conv2d', module=Quan_Conv2d)
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


    
    
    
    
    
    
    

# import math
# def getradix(xmax, bit_width):
#     if xmax == 0:
#         return 0
#     # elif np.isnan(xmax.detach()):
#     #     return 0
#     radix = bit_width - 1 - (math.floor(math.log2(xmax) + 1))
#     return radix

# class Quantizer(Function):
#     @staticmethod
#     def forward(ctx, input, radix):
#         scale = 2 ** radix
#         return torch.round(input * scale) / scale

#     @staticmethod
#     def backward(ctx, grad_output):
#         return grad_output, None


# def quantize(input, radix):
#     return Quantizer.apply(input, radix)

# def quan_w(w, nbit_w):
#     xmax = torch.max(torch.abs(w))
#     # w_min, w_max = get_percentile_min_max(w.view(-1), 100 - 0.1,
#     #                                                   0.1, output_tensor=True)
#     radix = getradix(xmax,nbit_w)
#     w = quantize(w,radix)
#     return w

# def quan_a(x, nbit_a):
#     xmax = torch.max(torch.abs(x))
#     radix = getradix(xmax,nbit_a)
#     scale = 2**radix-1
#     x = torch.round(x * scale) / scale
#     return x


           
# class Linear_Q(nn.Linear):
#     def __init__(self, in_features, out_features, bias=True, quan_name_w='dorefa', quan_name_a='dorefa', nbit_w=8, nbit_a=32):
#         super(Linear_Q, self).__init__(in_features, out_features, bias)
#         self.nbit_w = nbit_w
#         self.nbit_a = nbit_a
#         name_w_dict = {'dorefa': quan_w}
#         name_a_dict = {'dorefa': quan_a}
#         self.quan_w = name_w_dict[quan_name_w]
#         self.quan_a = name_a_dict[quan_name_a]
#     def forward(self, input):
#         if self.nbit_w < 32:
#             w = self.quan_w(self.weight, self.nbit_w)
#         else:
#             w = self.weight
#         if self.nbit_a < 32:
#             x = self.quan_a(input, self.nbit_a)
#         else:
#             x = input
#         output = F.linear(x, w, self.bias)
#         return output


# class QuanConv(nn.Conv2d):
#     """docstring for QuanConv"""
#     def __init__(self, in_channels, out_channels, kernel_size, quan_name_w='dorefa', quan_name_a='dorefa', nbit_w=4,
#                  nbit_a=32, stride=1,
#                  padding=0, dilation=1, groups=1,
#                  bias=True):
#         super(QuanConv, self).__init__(
#             in_channels, out_channels, kernel_size, stride, padding, dilation,
#             groups, bias)
#         self.nbit_w = nbit_w
#         self.nbit_a = nbit_a
#         name_w_dict = {'dorefa': quan_w}
#         name_a_dict = {'dorefa': quan_a}
#         self.quan_w = name_w_dict[quan_name_w]
#         self.quan_a = name_a_dict[quan_name_a]
#     def forward(self, input):
#         if self.nbit_w <=32:
#             w = self.quan_w(self.weight, self.nbit_w)
#         else:
#             w = self.weight
#         x = input
#         output = F.conv2d(x, w, self.bias, self.stride, self.padding, self.dilation, self.groups)
#         return output