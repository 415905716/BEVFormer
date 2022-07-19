import sys
from HAWQ.utils.models.q_resnet import Q_ResNet50_detr,Q_ResNet101_detr
from HAWQ.utils.models.q_transformer import * #Q_input_proj, Q_query_embed,QuantMultiheadAttention,Q_class_embed,Q_bbox_embed
from HAWQ.bit_config import *

class ARGs(object):
    def __init__(self):
        self.backbone = 'resnet50'
        self.quant_scheme = 'detr8w8a'
        self.bias_bit = 32
        self.channel_wise = True
        self.act_percentile = 0
        self.act_range_momentum = 0.99
        self.weight_percentile = 0
        self.fix_BN = True
        self.fix_BN_threshold = None
        self.checkpoint_iter = -1
        self.fixed_point_quantization = False

def letsquant(model,args=None):
    return letsquant_(model, args=args)

def letsquant_(model, args=None):
    if args is None:
        raise NotImplementedError
    quant_class_embed=args.quant_class_embed
    quant_bbox_embed=args.quant_bbox_embed
    quant_input_proj=args.quant_input_proj
    quant_backbone=args.quant_backbone
    quant_encoder=args.quant_encoder
    quant_decoder=args.quant_decoder
    transformer = getattr(model,'transformer')
    # quant others
    if quant_class_embed:
        class_embed = getattr(model,'class_embed')
        setattr(model,'class_embed',Q_class_embed(class_embed))
    if quant_bbox_embed:
        bbox_embed = getattr(model, 'bbox_embed')
        setattr(model,'bbox_embed', Q_bbox_embed(bbox_embed))
    if quant_encoder:
        # quant encoder
        encoder_layers = transformer.encoder.layers#getattr(transformer,'encoder.layers')
        for idx,module in enumerate(encoder_layers):
            Q_enconder_layer = Q_TransformerEncoderLayer(module)
            setattr(model.transformer.encoder.layers,f"{idx}",Q_enconder_layer)
    if quant_decoder:
        # quant decoder
        decoder_layers = transformer.decoder.layers
        for idx,module in enumerate(decoder_layers):
            Q_decoder_layer = Q_TransformerDecoderLayer(module)
            setattr(model.transformer.decoder.layers,f"{idx}",Q_decoder_layer)
    if quant_backbone:
        # quant backbone
        backbone_without_pos = getattr(model,'backbone')[0]
        if quant_backbone:
            if args.backbone=='resnet101':
                Q_backbone_without_pos = Q_ResNet101_detr(backbone_without_pos)
            else:
                Q_backbone_without_pos = Q_ResNet50_detr(backbone_without_pos)
            setattr(model.backbone, '0', Q_backbone_without_pos)
    if quant_input_proj:
        # quant input_proj
        input_proj = getattr(model, 'input_proj')
        q_input_proj = Q_input_proj(input_proj)
        setattr(model,'input_proj',q_input_proj)

    # after setup q_model, set quant config
    print("-"*80)
    bit_config = bit_config_dict["bit_config_" + args.backbone + "_" + args.quant_scheme]
    print("Load configuration:","bit_config_" + args.backbone + "_" + args.quant_scheme)
    print("bit_config:",bit_config)
    print("-"*80)
    name_counter = 0
    for name, m in model.named_modules():
        need_quant = False
        if name.startswith("transformer.encoder") and "quant" in name:
            need_quant = True
        if name.startswith("transformer.decoder") and "quant" in name:
            need_quant = True
        if name.startswith("backbone"):
            real_name = name
            name = name.replace("backbone.0.","")
        else:
            real_name = name
        if name in bit_config.keys() or need_quant:
            print("[QUANT]",real_name)
            name_counter += 1
            setattr(m, 'quant_mode', 'symmetric')
            setattr(m, 'bias_bit', args.bias_bit)
            setattr(m, 'quantize_bias', (args.bias_bit != 0))
            setattr(m, 'per_channel', args.channel_wise)
            setattr(m, 'act_percentile', args.act_percentile)
            setattr(m, 'act_range_momentum', args.act_range_momentum)
            setattr(m, 'weight_percentile', args.weight_percentile)
            setattr(m, 'fix_flag', False)
            setattr(m, 'fix_BN', args.fix_BN)
            setattr(m, 'fix_BN_threshold', args.fix_BN_threshold)
            setattr(m, 'training_BN_mode', args.fix_BN)
            setattr(m, 'checkpoint_iter_threshold', args.checkpoint_iter)
            setattr(m, 'fixed_point_quantization', args.fixed_point_quantization)

            if type(bit_config.get(name,None)) is tuple:
                bitwidth = bit_config[name][0]
                if bit_config[name][1] == 'hook':
                    # m.register_forward_hook(hook_fn_forward)
                    global hook_keys
                    hook_keys.append(name)
            else:
                bitwidth = bit_config.get(name,8)

            if hasattr(m, 'activation_bit'):
                setattr(m, 'activation_bit', bitwidth)
                if bitwidth == 4:
                    setattr(m, 'quant_mode', 'asymmetric')
            else:
                setattr(m, 'weight_bit', bitwidth)
        else:
            print('[FP==>]',real_name)
    print(model)
    print("-"*80)
    return model
