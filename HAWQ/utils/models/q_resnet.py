"""
    Quantized ResNet for ImageNet-1K, implemented in PyTorch.
    Original paper: 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
"""

import torch
import torch.nn as nn
import copy
from ..quantization_utils.quant_modules import *
from pytorchcv.models.common import ConvBlock
from pytorchcv.models.shufflenetv2 import ShuffleUnit, ShuffleInitBlock
import time
import logging
import sys
sys.path.append('../../../detr')
from util.misc import NestedTensor, is_main_process

class Q_ResNet18_detr(nn.Module):
    """
        Quantized backbone(ResNet18) model of detr.
    """
    def __init__(self, model):
        super().__init__()

        body = getattr(model, 'body')

        self.quant_input = QuantAct()
        self.quant_init_block_convbn = QuantBnConv2d()
        self.quant_init_block_convbn.set_param(body.conv1, body.bn1)

        self.quant_act_int32 = QuantAct()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [2, 2, 2, 2]

        for layer_num in range(0, 4):
            layer = getattr(body, "layer{}".format(layer_num + 1))
            for unit_num in range(0, self.channel[layer_num]):
                unit = getattr(layer, "{}".format(unit_num))
                quant_unit = Q_ResBlockBn_detr()
                quant_unit.set_param(unit)
                setattr(self, f"stage{layer_num + 1}.unit{unit_num + 1}", quant_unit)

    def forward(self, x):
        # print('type(x):',type(x))
        tensors, mask = x.decompose()
        x = tensors
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)

        x = self.act(x)
        # print("X:",x)
        # print(x/act_scaling_factor.cuda())
        # raise NotImplementedError

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)
        # mask
        m = mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        x = NestedTensor((x,act_scaling_factor), mask)
        return x


class Q_ResNet18(nn.Module):
    """
        Quantized ResNet18 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, model):
        super().__init__()
        features = getattr(model, 'features')
        init_block = getattr(features, 'init_block')

        self.quant_input = QuantAct()

        self.quant_init_block_convbn = QuantBnConv2d()
        self.quant_init_block_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

        self.quant_act_int32 = QuantAct()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [2, 2, 2, 2]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResBlockBn()
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_block_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x

class Q_ResNet50_detr(nn.Module):
    """
        Quantized backbone(ResNet50) model of detr.
    """
    def __init__(self, model):
        super().__init__()

        body = getattr(model, 'body')

        self.quant_input = QuantAct()
        self.quant_init_convbn = QuantBnConv2d()
        self.quant_init_convbn.set_param(body.conv1, body.bn1)

        self.quant_act_int32 = QuantAct()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 6, 3]

        for layer_num in range(0, 4):
            layer = getattr(body, "layer{}".format(layer_num + 1))
            for unit_num in range(0, self.channel[layer_num]):
                unit = getattr(layer, "{}".format(unit_num))
                quant_unit = Q_ResUnitBn_detr()
                quant_unit.set_param(unit)
                setattr(self, f"stage{layer_num + 1}.unit{unit_num + 1}", quant_unit)

    def forward(self, x):
        # print('type(x):',type(x))
        tensors, mask = x.decompose()
        x = tensors
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)

        x = self.act(x)
        # print("X:",x)
        # print(x/act_scaling_factor.cuda())
        # raise NotImplementedError

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)
        # mask
        m = mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        x = NestedTensor((x,act_scaling_factor), mask)
        return x

    
class Q_ResNet50(nn.Module):
    """
        Quantized ResNet50 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, model):
        super().__init__()

        features = getattr(model, 'features')

        init_block = getattr(features, 'init_block')
        self.quant_input = QuantAct()
        self.quant_init_convbn = QuantBnConv2d()
        self.quant_init_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

        self.quant_act_int32 = QuantAct()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 6, 3]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResUnitBn()
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x

class Q_ResNet101_detr(nn.Module):
    """
        Quantized backbone(ResNet101) model of detr.
    """
    def __init__(self, model):
        super().__init__()

        body = getattr(model, 'body')

        self.quant_input = QuantAct()
        self.quant_init_convbn = QuantBnConv2d()
        self.quant_init_convbn.set_param(body.conv1, body.bn1)

        self.quant_act_int32 = QuantAct()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 23, 3]

        for layer_num in range(0, 4):
            layer = getattr(body, "layer{}".format(layer_num + 1))
            for unit_num in range(0, self.channel[layer_num]):
                unit = getattr(layer, "{}".format(unit_num))
                quant_unit = Q_ResUnitBn_detr()
                quant_unit.set_param(unit)
                setattr(self, f"stage{layer_num + 1}.unit{unit_num + 1}", quant_unit)

    def forward(self, x):
        tensors, mask = x.decompose()
        x = tensors
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)
        # mask
        m = mask
        assert m is not None
        mask = F.interpolate(m[None].float(), size=x.shape[-2:]).to(torch.bool)[0]
        x = NestedTensor((x,act_scaling_factor), mask)
        return x

class Q_ResNet101(nn.Module):
    """
       Quantized ResNet101 model from 'Deep Residual Learning for Image Recognition,' https://arxiv.org/abs/1512.03385.
    """
    def __init__(self, model):
        super().__init__()

        features = getattr(model, 'features')

        init_block = getattr(features, 'init_block')
        self.quant_input = QuantAct()
        self.quant_init_convbn = QuantBnConv2d()
        self.quant_init_convbn.set_param(init_block.conv.conv, init_block.conv.bn)

        self.quant_act_int32 = QuantAct()

        self.pool = nn.MaxPool2d(kernel_size=3, stride=2, padding=1)
        self.act = nn.ReLU()

        self.channel = [3, 4, 23, 3]

        for stage_num in range(0, 4):
            stage = getattr(features, "stage{}".format(stage_num + 1))
            for unit_num in range(0, self.channel[stage_num]):
                unit = getattr(stage, "unit{}".format(unit_num + 1))
                quant_unit = Q_ResUnitBn()
                quant_unit.set_param(unit)
                setattr(self, f"stage{stage_num + 1}.unit{unit_num + 1}", quant_unit)

        self.final_pool = QuantAveragePool2d(kernel_size=7, stride=1)

        self.quant_act_output = QuantAct()

        output = getattr(model, 'output')
        self.quant_output = QuantLinear()
        self.quant_output.set_param(output)

    def forward(self, x):
        x, act_scaling_factor = self.quant_input(x)

        x, weight_scaling_factor = self.quant_init_convbn(x, act_scaling_factor)

        x = self.pool(x)
        x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, None, None)

        x = self.act(x)

        for stage_num in range(0, 4):
            for unit_num in range(0, self.channel[stage_num]):
                tmp_func = getattr(self, f"stage{stage_num+1}.unit{unit_num+1}")
                x, act_scaling_factor = tmp_func(x, act_scaling_factor)

        x = self.final_pool(x, act_scaling_factor)

        x, act_scaling_factor = self.quant_act_output(x, act_scaling_factor)
        x = x.view(x.size(0), -1)
        x = self.quant_output(x, act_scaling_factor)

        return x


class Q_ResUnitBn(nn.Module):
    """
       Quantized ResNet unit with residual path.
    """
    def __init__(self):
        super(Q_ResUnitBn, self).__init__()

    def set_param(self, unit):
        self.resize_identity = unit.resize_identity

        self.quant_act = QuantAct()

        convbn1 = unit.body.conv1
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)
        self.quant_act1 = QuantAct()

        convbn2 = unit.body.conv2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)
        self.quant_act2 = QuantAct()

        convbn3 = unit.body.conv3
        self.quant_convbn3 = QuantBnConv2d()
        self.quant_convbn3.set_param(convbn3.conv, convbn3.bn)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn3(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None)

        x = nn.ReLU()(x)

        return x, act_scaling_factor

class Q_ResUnitBn_detr(nn.Module):
    """
       Quantized ResNet unit with residual path.
    """
    def __init__(self):
        super(Q_ResUnitBn_detr, self).__init__()

    def set_param(self, unit):
        self.resize_identity = (unit.__dict__.get("downsample",True) is True)

        self.quant_act = QuantAct()

        convbn1 = unit
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv1, convbn1.bn1)
        self.quant_act1 = QuantAct()

        convbn2 = unit
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv2, convbn2.bn2)
        self.quant_act2 = QuantAct()

        convbn3 = unit
        self.quant_convbn3 = QuantBnConv2d()
        self.quant_convbn3.set_param(convbn3.conv3, convbn3.bn3)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(getattr(unit.downsample,"0"), getattr(unit.downsample,"1"))

        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act2(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn3(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None)

        x = nn.ReLU()(x)

        return x, act_scaling_factor

class Q_ResBlockBn_detr(nn.Module):
    """
        Quantized ResNet block with residual path.
    """
    def __init__(self):
        super(Q_ResBlockBn_detr, self).__init__()

    def set_param(self, unit):
        self.resize_identity = (unit.__dict__.get("downsample",True) is True)

        self.quant_act = QuantAct()

        convbn1 = unit
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv1, convbn1.bn1)
        self.quant_act1 = QuantAct()

        convbn2 = unit
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv2, convbn2.bn2)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(getattr(unit.downsample,"0"), getattr(unit.downsample,"1"))

        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None)

        x = nn.ReLU()(x)

        return x, act_scaling_factor
    
class Q_ResBlockBn(nn.Module):
    """
        Quantized ResNet block with residual path.
    """
    def __init__(self):
        super(Q_ResBlockBn, self).__init__()

    def set_param(self, unit):
        self.resize_identity = unit.resize_identity

        self.quant_act = QuantAct()

        convbn1 = unit.body.conv1
        self.quant_convbn1 = QuantBnConv2d()
        self.quant_convbn1.set_param(convbn1.conv, convbn1.bn)

        self.quant_act1 = QuantAct()

        convbn2 = unit.body.conv2
        self.quant_convbn2 = QuantBnConv2d()
        self.quant_convbn2.set_param(convbn2.conv, convbn2.bn)

        if self.resize_identity:
            self.quant_identity_convbn = QuantBnConv2d()
            self.quant_identity_convbn.set_param(unit.identity_conv.conv, unit.identity_conv.bn)

        self.quant_act_int32 = QuantAct()

    def forward(self, x, scaling_factor_int32=None):
        # forward using the quantized modules
        if self.resize_identity:
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)
            identity_act_scaling_factor = act_scaling_factor.clone()
            identity, identity_weight_scaling_factor = self.quant_identity_convbn(x, act_scaling_factor)
        else:
            identity = x
            x, act_scaling_factor = self.quant_act(x, scaling_factor_int32)

        x, weight_scaling_factor = self.quant_convbn1(x, act_scaling_factor)
        x = nn.ReLU()(x)
        x, act_scaling_factor = self.quant_act1(x, act_scaling_factor, weight_scaling_factor)

        x, weight_scaling_factor = self.quant_convbn2(x, act_scaling_factor)

        x = x + identity

        if self.resize_identity:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, identity_act_scaling_factor, identity_weight_scaling_factor)
        else:
            x, act_scaling_factor = self.quant_act_int32(x, act_scaling_factor, weight_scaling_factor, identity, scaling_factor_int32, None)

        x = nn.ReLU()(x)

        return x, act_scaling_factor


def q_resnet18(model):
    net = Q_ResNet18(model)
    return net


def q_resnet50(model):
    net = Q_ResNet50(model)
    return net


def q_resnet101(model):
    net = Q_ResNet101(model)
    return net
