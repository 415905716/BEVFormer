"""
Quantize Transformer
"""
import torch
import torch.nn as nn
import torch.nn.functional as F
import copy
from ..quantization_utils.quant_modules import *
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock
import time
import logging
import sys
sys.path.append('../../../detr')
from util.misc import NestedTensor, is_main_process

class Q_input_proj(nn.Module):
    def __init__(self, model):
        super().__init__()
        self.quant_conv = QuantConv2d()
        self.quant_conv.set_param(model)
        self.quant_act_int = QuantAct()
        print("quant_act_int activate!")
    def forward(self, x, pre_act_scaling_factor=None):
        # x, weight_scaling_factor = self.quant_conv(x,pre_act_scaling_factor)
        # x, act_scaling_factor = self.quant_act_int(x, pre_act_scaling_factor, weight_scaling_factor)
        # return (x, act_scaling_factor)
        return self.quant_conv(x,pre_act_scaling_factor)

class Q_query_embed(nn.Module):
    def __init__(self, emb):
        super().__init__()
        self.quant_emb = QuantAct()
    def forward(self,x):
        return self.quant_emb(x)

class QuantLayerNorm(nn.Module):
    def __init__(self,
                 weight_bit=4,
                 full_precision_flag=False,
                 quant_mode="symmetric",
                 per_channel=False,
                 fix_flag=False,
                 weight_percentile=0,
                 fix_LN=False,
                 fix_LN_threshold=None):
        super(QuantLayerNorm, self).__init__()
        self.weight_bit = weight_bit
        self.full_precision_flag = full_precision_flag
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.quant_mode = quant_mode
        self.fix_LN = fix_LN
        self.training_LN_mode = fix_LN
        self.fix_LN_threshold = fix_LN_threshold
        self.counter = 1
    def set_param(self, layernorm):
        self.normalized_shape = layernorm.normalized_shape
        self.register_buffer('ln_scaling_factor', torch.zeros(self.normalized_shape))
        self.register_buffer('weight_integer', torch.zeros_like(layernorm.weight.data))
        self.register_buffer('bias_integer', torch.zeros_like(layernorm.bias))

        self.ln = layernorm
        # self.ln.eps = 1e-5
    # TODO:
    def __repr__(self):
        conv_s = super(QuantLayerNorm, self).__repr__()
        s = "({0}, weight_bit={1}, bias_bit={2}, groups={3}, wt-channel-wise={4}, wt-percentile={5}, quant_mode={6})".format(
            conv_s, self.weight_bit, '?', '?', self.per_channel, self.weight_percentile, self.quant_mode)
        return s
    def fix(self):
        """
        fix the LN statistics by setting fix_LN to True
        """
        self.fix_flag = True
        self.fix_LN = True

    def unfix(self):
        """
        change the mode (fixed or not) of LN statistics to its original status
        """
        self.fix_flag = False
        self.fix_LN = self.training_LN_mode
    def forward(self, x, pre_act_scaling_factor=None):
        """
        x: the input activation
        pre_act_scaling_factor: the scaling factor of the previous activation quantization layer

        """
        if type(x) is tuple:
            pre_act_scaling_factor = x[1]
            x = x[0]

        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        # determine whether to fold LN or not
        if self.fix_flag == False:
            self.counter += 1
            if (self.fix_LN_threshold == None) or (self.counter < self.fix_LN_threshold):
                self.fix_LN = self.training_LN_mode
            else:
                if self.counter == self.fix_LN_threshold:
                    print("Start Training with Folded LN")
                self.fix_LN = True

        # run the forward without folding LN
        if self.fix_LN == False:
            layer_mean = torch.mean(x.data,dim=-1,keepdim=True)
            layer_var = torch.var(x.data,dim=-1,keepdim=True)
            (x - layer_mean)/torch.sqrt(layer_var)
            # w_transform = self.conv.weight.data.contiguous().view(self.conv.out_channels, -1)
            # w_min = w_transform.min(dim=1).values
            # w_max = w_transform.max(dim=1).values
            #
            # conv_scaling_factor = symmetric_linear_quantization_params(self.weight_bit, w_min, w_max, self.per_channel)
            # weight_integer = self.weight_function(self.conv.weight, self.weight_bit, conv_scaling_factor)
            # conv_output = F.conv2d(x, weight_integer, self.conv.bias, self.conv.stride, self.conv.padding,
            #                        self.conv.dilation, self.conv.groups) * conv_scaling_factor.view(1, -1, 1, 1)
            #
            # batch_mean = torch.mean(conv_output, dim=(0, 2, 3))
            # batch_var = torch.var(conv_output, dim=(0, 2, 3))
            #
            # # update mean and variance in running stats
            # self.bn.running_mean = self.bn.running_mean.detach() * self.bn.momentum + (
            #             1 - self.bn.momentum) * batch_mean
            # self.bn.running_var = self.bn.running_var.detach() * self.bn.momentum + (1 - self.bn.momentum) * batch_var
            #
            # output_factor = self.bn.weight.view(1, -1, 1, 1) / torch.sqrt(batch_var + self.bn.eps).view(1, -1, 1, 1)
            # output = output_factor * (conv_output - batch_mean.view(1, -1, 1, 1)) + self.bn.bias.view(1, -1, 1, 1)
            #
            # return (output, conv_scaling_factor.view(-1) * output_factor.view(-1))
        # fix LN, weights and bias stop updating during training
        else:
            layer_mean = torch.mean(x.data,dim=-1,keepdim=True)
            layer_var = torch.var(x.data,dim=-1,keepdim=True)


            #-------------------
            running_std = torch.sqrt(self.bn.running_var.detach() + self.bn.eps)
            scale_factor = self.bn.weight / running_std
            scaled_weight = self.conv.weight * scale_factor.reshape([self.conv.out_channels, 1, 1, 1])

            if self.conv.bias is not None:
                scaled_bias = self.conv.bias
            else:
                scaled_bias = torch.zeros_like(self.bn.running_mean)
            scaled_bias = (scaled_bias - self.bn.running_mean.detach()) * scale_factor + self.bn.bias

            if not self.full_precision_flag:
                if self.per_channel:
                    w_transform = scaled_weight.data.contiguous().view(self.conv.out_channels, -1)

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
                else:
                    if self.weight_percentile == 0:
                        w_min = scaled_weight.data.min()
                        w_max = scaled_weight.data.max()
                    else:
                        w_min, w_max = get_percentile_min_max(scaled_weight.view(-1), 100 - self.weight_percentile,
                                                              self.weight_percentile, output_tensor=True)

                if self.quant_mode == 'symmetric':
                    self.convbn_scaling_factor = symmetric_linear_quantization_params(self.weight_bit,
                                                                                      w_min, w_max, self.per_channel).cuda()
                    self.weight_integer = self.weight_function(scaled_weight, self.weight_bit,
                                                               self.convbn_scaling_factor)
                    if self.quantize_bias:
                        bias_scaling_factor = self.convbn_scaling_factor.view(1, -1) * pre_act_scaling_factor.view(1,-1)
                        self.bias_integer = self.weight_function(scaled_bias, self.bias_bit, bias_scaling_factor)
                    self.convbn_scaled_bias = scaled_bias
                else:
                    raise Exception('For weight, we only support symmetric quantization.')

            pre_act_scaling_factor = pre_act_scaling_factor.view(1, -1, 1, 1)
            x_int = x / pre_act_scaling_factor
            correct_output_scale = bias_scaling_factor.view(1, -1, 1, 1)

            return (F.conv2d(x_int, self.weight_integer, self.bias_integer, self.conv.stride, self.conv.padding,
                             self.conv.dilation, self.conv.groups) * correct_output_scale, self.convbn_scaling_factor)
class InProjector(object):
    def __init__(self,weight,bias,in_features="neveruse",out_features='neveruse'):
        self.weight = weight
        self.bias = bias
        self.in_features = in_features
        self.out_features = out_features


class QuantMultiheadAttention(nn.Module):
    """
    borrow codes from HAWQ
    """
    def __init__(self,
                 weight_bit=4,
                 bias_bit=None,
                 full_precision_flag=False,
                 quant_mode="symmetric",
                 per_channel=False,
                 fix_flag=False,
                 weight_percentile=0):
        super(QuantMultiheadAttention, self).__init__()
        self.full_precision_flag = full_precision_flag
        self.weight_bit = weight_bit
        self.quant_mode = quant_mode
        self.per_channel = per_channel
        self.fix_flag = fix_flag
        self.weight_percentile = weight_percentile
        self.bias_bit = bias_bit
        self.quantize_bias = (False if bias_bit is None else True)
        self.in_proj_q = QuantLinear()
        self.in_proj_k = QuantLinear()
        self.in_proj_v = QuantLinear()
        self.out_proj = QuantLinear()
        self.quant_query = QuantAct()
        self.quant_key = QuantAct()
        self.quant_value = QuantAct()
        self.quant_query_aft_proj = QuantAct()
        self.quant_key_aft_proj = QuantAct()
        self.quant_value_aft_proj = QuantAct()
        self.quant_attn_output_weights_aft_softmax = QuantAct()
        self.quant_attn_output_before_out_proj = QuantAct()

    def set_param(self, MHSA):
        self.embed_dim = MHSA.embed_dim
        self.num_heads = MHSA.num_heads
        self.dropout = MHSA.dropout
        self.head_dim = MHSA.head_dim

        qProjector,kProjector,vProjector = self.assemble_qkv_projector(in_proj_weight=getattr(MHSA,'in_proj_weight'),in_proj_bias=getattr(MHSA,'in_proj_bias'))
        self.in_proj_q.set_param(qProjector)
        self.in_proj_k.set_param(kProjector)
        self.in_proj_v.set_param(vProjector)
        self.out_proj.set_param(getattr(MHSA,'out_proj'))
        # self.register_buffer('in_proj_weight_scaling_factor', torch.zeros(1)) ## consider other settings?
        # self.in_proj_weight = Parameter(MHSA.in_proj_weight.data.clone())
        # self.register_buffer('in_proj_weight_integer', torch.zeros_like(self.in_proj_weight, dtype=torch.int8))
        # self.register_buffer('in_proj_bias_scaling_factor', torch.zeros(1))
        # self.in_proj_bias = Parameter(MHSA.in_proj_bias.data.clone())
        # self.register_buffer('in_proj_bias_integer', torch.zeros_like(self.in_proj_weight, dtype=torch.int8))

    def assemble_qkv_projector(self,in_proj_weight,in_proj_bias):
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = 0
        _end = self.embed_dim
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        qProjector = InProjector(_w,_b,self.embed_dim,self.embed_dim)
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = self.embed_dim
        _end = self.embed_dim * 2
        _w = in_proj_weight[_start:_end, :]
        if _b is not None:
            _b = _b[_start:_end]
        kProjector = InProjector(_w,_b,self.embed_dim,self.embed_dim)
        # This is inline in_proj function with in_proj_weight and in_proj_bias
        _b = in_proj_bias
        _start = self.embed_dim * 2
        _end = None
        _w = in_proj_weight[_start:, :]
        if _b is not None:
            _b = _b[_start:]
        vProjector = InProjector(_w,_b,self.embed_dim,self.embed_dim)
        return (qProjector,kProjector,vProjector)
    def forward(self, query, key, value,

                key_padding_mask=None,
                need_weights=True, attn_mask=None, pre_act_scaling_factor=None):
        # if type(x) is tuple:
        #     pre_act_scaling_factor = x[1]
        #     x = x[0]
        # pre_act_scaling_factor = pre_act_scaling_factor.cuda()
        if self.quant_mode == "symmetric":
            self.weight_function = SymmetricQuantFunction.apply
        elif self.quant_mode == "asymmetric":
            self.weight_function = AsymmetricQuantFunction.apply
        else:
            raise ValueError("unknown quant mode: {}".format(self.quant_mode))

        return multi_head_attention_forward_SLIM(self,
                query, key, value, embed_dim_to_check=self.embed_dim, num_heads=self.num_heads,
                #in_proj_weight=self.in_proj_weight, in_proj_bias=self.in_proj_bias,
                bias_k=None, bias_v=None, add_zero_attn=False,
                dropout_p=self.dropout, #out_proj_weight=self.out_proj.weight, out_proj_bias=self.out_proj.bias,
                training=self.training,
                key_padding_mask=key_padding_mask, need_weights=need_weights,
                attn_mask=attn_mask)

def multi_head_attention_forward_SLIM(self_arg,
                                 query,                           # type: Tensor
                                 key,                             # type: Tensor
                                 value,                           # type: Tensor
                                 embed_dim_to_check,              # type: int
                                 num_heads,                       # type: int
                                 # in_proj_weight,                  # type: Tensor
                                 # in_proj_bias,                    # type: Tensor
                                 bias_k,                          # type: Optional[Tensor]
                                 bias_v,                          # type: Optional[Tensor]
                                 add_zero_attn,                   # type: bool
                                 dropout_p,                       # type: float
                                 # out_proj_weight,                 # type: Tensor
                                 # out_proj_bias,                   # type: Tensor
                                 training=True,                   # type: bool
                                 key_padding_mask=None,           # type: Optional[Tensor]
                                 need_weights=True,               # type: bool
                                 attn_mask=None,                  # type: Optional[Tensor]
                                 use_separate_proj_weight=False,  # type: bool
                                 q_proj_weight=None,              # type: Optional[Tensor]
                                 k_proj_weight=None,              # type: Optional[Tensor]
                                 v_proj_weight=None,              # type: Optional[Tensor]
                                 static_k=None,                   # type: Optional[Tensor]
                                 static_v=None                    # type: Optional[Tensor]
                                 ):
    """
    pytorch source code with modification
    """

    # if not torch.jit.is_scripting():
    #     tens_ops = (query, key, value)
    #     if any([type(t) is not Tensor for t in tens_ops]) and has_torch_function(tens_ops):
    #         raise NotImplementedError
    tgt_len, bsz, embed_dim = query.size()
    assert embed_dim == embed_dim_to_check
    assert key.size() == value.size()

    head_dim = embed_dim // num_heads
    assert head_dim * num_heads == embed_dim, "embed_dim must be divisible by num_heads"
    scaling = float(head_dim) ** -0.5

    if not use_separate_proj_weight:
        if torch.equal(query, key) and torch.equal(key, value):
            raise NotImplementedError
        elif torch.equal(key, value):
            raise NotImplementedError
        else:
            # This is inline in_proj function with in_proj_weight and in_proj_bias
            query, act_scaling_factor = self_arg.quant_query(query)
            # print(query)
            # print(query/act_scaling_factor)
            q = self_arg.in_proj_q(query, act_scaling_factor)
            # print(q)
            # raise NotImplementedError
            # 结果一致

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            key, act_scaling_factor = self_arg.quant_key(key)
            k = self_arg.in_proj_k(key, act_scaling_factor)

            # This is inline in_proj function with in_proj_weight and in_proj_bias
            value, act_scaling_factor = self_arg.quant_value(value)
            v = self_arg.in_proj_v(value, act_scaling_factor)
            # print(q)
            # print(k)
            # print(v)
            # raise NotImplementedError
    else:
        raise NotImplementedError
    q = q * scaling

    if attn_mask is not None:
        if attn_mask.dim() == 2:
            attn_mask = attn_mask.unsqueeze(0)
            if list(attn_mask.size()) != [1, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 2D attn_mask is not correct.')
        elif attn_mask.dim() == 3:
            if list(attn_mask.size()) != [bsz * num_heads, query.size(0), key.size(0)]:
                raise RuntimeError('The size of the 3D attn_mask is not correct.')
        else:
            raise RuntimeError("attn_mask's dimension {} is not supported".format(attn_mask.dim()))
        # attn_mask's dim is 3 now.

    if bias_k is not None and bias_v is not None:
        raise NotImplementedError
    else:
        assert bias_k is None
        assert bias_v is None

    q = q.contiguous().view(tgt_len, bsz * num_heads, head_dim).transpose(0, 1)
    if k is not None:
        k = k.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)
    if v is not None:
        v = v.contiguous().view(-1, bsz * num_heads, head_dim).transpose(0, 1)

    if static_k is not None:
        raise NotImplementedError

    if static_v is not None:
        raise NotImplementedError

    src_len = k.size(1)

    if key_padding_mask is not None:
        assert key_padding_mask.size(0) == bsz
        assert key_padding_mask.size(1) == src_len

    if add_zero_attn:
        raise NotImplementedError

    q, q_act_scaling_factor = self_arg.quant_query_aft_proj(q)
    q_int = q/q_act_scaling_factor
    k, k_act_scaling_factor = self_arg.quant_key_aft_proj(k)
    k_int = k/k_act_scaling_factor
    attn_output_weights = torch.bmm(q_int, k_int.transpose(1, 2))*q_act_scaling_factor*k_act_scaling_factor
    assert list(attn_output_weights.size()) == [bsz * num_heads, tgt_len, src_len]

    if attn_mask is not None:
        raise NotImplementedError
        # attn_output_weights += attn_mask

    if key_padding_mask is not None:
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        attn_output_weights = attn_output_weights.masked_fill(
            key_padding_mask.unsqueeze(1).unsqueeze(2),
            float('-inf'),
        )
        attn_output_weights = attn_output_weights.view(bsz * num_heads, tgt_len, src_len)

    attn_output_weights = F.softmax(
        attn_output_weights, dim=-1)
    attn_output_weights = F.dropout(attn_output_weights, p=dropout_p, training=training)

    v, v_act_scaling_factor = self_arg.quant_value_aft_proj(v)
    attn_output_weights, attn_act_scaling_factor = self_arg.quant_attn_output_weights_aft_softmax(attn_output_weights)
    attn_output = torch.bmm(attn_output_weights/attn_act_scaling_factor,
                        v/v_act_scaling_factor) *v_act_scaling_factor*attn_act_scaling_factor
    assert list(attn_output.size()) == [bsz * num_heads, tgt_len, head_dim]
    attn_output = attn_output.transpose(0, 1).contiguous().view(tgt_len, bsz, embed_dim)
    # attn_output = linear(attn_output, out_proj_weight, out_proj_bias)
    attn_output,act_scaling_factor = self_arg.quant_attn_output_before_out_proj(attn_output)
    attn_output = self_arg.out_proj(attn_output,act_scaling_factor)

    if need_weights:
        # average attention weights over heads
        attn_output_weights = attn_output_weights.view(bsz, num_heads, tgt_len, src_len)
        return attn_output, attn_output_weights.sum(dim=1) / num_heads
    else:
        return attn_output, None




class Q_TransformerEncoderLayer(nn.Module):
    def __init__(self, model):
        super().__init__()
        self_attn = getattr(model,"self_attn")
        linear1 = getattr(model, 'linear1')
        self.quant_self_attn = (None)
        self.linear1 = QuantLinear()
        self.linear1.set_param()
        dropout = nn.Dropout(p)
        linear2 = getattr(model, 'linear2')
        self.linear1 = QuantLinear()
        self.linear1.set_param()
        self.quant_act1 = QuantAct()
        self.norm1 = QuantLayerNorm()
        self.quant_act2 = QuantAct()
        self.norm2 = QuantLayerNorm()
        dropout1 = nn.Dropout(p)
        dropout2 = nn.Dropout(p)
    def forward(self,x):
        pass
