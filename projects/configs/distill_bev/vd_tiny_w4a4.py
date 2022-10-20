# BEvFormer-tiny consumes at lease 6700M GPU memory
# compared to bevformer_base, bevformer_tiny has
# smaller backbone: R101-DCN -> R50
# smaller BEV: 200*200 -> 50*50
# less encoder layers: 6 -> 3
# smaller input size: 1600*900 -> 800*450
# multi-scale feautres -> single scale features (C5)


_base_ = [
    '../datasets/custom_nus-3d.py',
    '../_base_/default_runtime.py'
]
#
plugin = True
plugin_dir = 'projects/mmdet3d_plugin/'

# If point cloud range is changed, the models should also change their point
# cloud range accordingly
point_cloud_range = [-51.2, -51.2, -5.0, 51.2, 51.2, 3.0]
voxel_size = [0.2, 0.2, 8]




img_norm_cfg = dict(
    mean=[123.675, 116.28, 103.53], std=[58.395, 57.12, 57.375], to_rgb=True)

# For nuScenes we usually do 10-class detection
class_names = [
    'car', 'truck', 'construction_vehicle', 'bus', 'trailer', 'barrier',
    'motorcycle', 'bicycle', 'pedestrian', 'traffic_cone'
]

input_modality = dict(
    use_lidar=False,
    use_camera=True,
    use_radar=False,
    use_map=False,
    use_external=True)

_dim_ = 256
_pos_dim_ = _dim_//2
_ffn_dim_ = _dim_*2
_num_levels_ = 1
bev_h_ = 50
bev_w_ = 50
queue_length = 3 # each sequence contains `queue_length` frames.

full_precision_bit=32
quantize_backbone_bit=4
quantize_bit=32
checkpoint_t='/home/niko/BEVFormer/ckpts/bevformer_tiny_epoch_24.pth'
teacher = dict(
    type='mmdet.BEVFormer1',
    use_grid_mask=True,
    video_test_mode=True,
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_t),
    # pretrained=dict(img='/home/niko/BEVFormer/ckpts/resnet50_uniform4/checkpoint.pth.tar'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        conv_cfg=dict(type='Conv2d',weight_bit=quantize_bit, activation_bit=quantize_bit, weight_percentile=0, act_percentile=0, full_precision_flag=True, quant_act=False),
        # conv1_cfg=dict(type='Conv2d',weight_bit=8, activation_bit=8, weight_percentile=0, act_percentile=0, full_precision_flag=False, quant_act=True),
        # init_cfg=dict(type='Pretrained', checkpoint='/home/niko/BEVFormer/ckpts/resnet50_uniform4/checkpoint.pth.tar'),
        ),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True,
        conv_cfg=dict(type='Conv2d', weight_bit=quantize_bit, activation_bit=quantize_bit, full_precision_flag=False, quant_act=True)),
    pts_bbox_head=dict(
        type='BEVFormerHead', 
        bev_h=bev_h_,
        bev_w=bev_w_,
        weight_bit=quantize_bit,
        activation_bit=quantize_bit,#quant_linear_Q
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer', 
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            weight_bit=quantize_bit,
            activation_bit=quantize_bit,#quant_linear_Q
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    weight_bit=quantize_bit,
                    activation_bit=quantize_bit,  #quant_linear_Q
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1,
                            weight_bit=quantize_bit,
                            activation_bit=quantize_bit,  #quant_linear_Q
                            ),
                        dict(
                            type='SpatialCrossAttention', 
                            pc_range=point_cloud_range,
                            weight_bit=quantize_bit,
                            activation_bit=quantize_bit,#quant_linear_Q
                            deformable_attention=dict(
                                type='MSDeformableAttention3D', 
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                                weight_bit=quantize_bit,
                                activation_bit=quantize_bit,),#quant_linear_Q
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention', 
                            embed_dims=_dim_,
                            num_levels=1,
                            weight_bit=quantize_bit,
                            activation_bit=quantize_bit, #quant_linear_Q
                            ),
                    ],
                    weight_bit=quantize_bit,
                    activation_bit=quantize_bit, #quant_linear_Q
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))
checkpoint_s='/home/niko/BEVFormer/ckpts/bevformer_tiny_epoch_24.pth'
student = dict(
    type='mmdet.BEVFormer1',
    use_grid_mask=True,
    video_test_mode=True,
    init_cfg=dict(type='Pretrained', checkpoint=checkpoint_s),
    # pretrained=dict(img='/home/niko/BEVFormer/ckpts/resnet50_uniform4/checkpoint.pth.tar'),
    img_backbone=dict(
        type='ResNet',
        depth=50,
        num_stages=4,
        out_indices=(3,),
        frozen_stages=1,
        norm_cfg=dict(type='BN', requires_grad=False),
        norm_eval=True,
        style='pytorch',
        conv_cfg=dict(type='Conv2d',weight_bit=quantize_backbone_bit, activation_bit=quantize_backbone_bit, weight_percentile=0, act_percentile=0, full_precision_flag=False, quant_act=True),
        conv1_cfg=dict(type='Conv2d',weight_bit=8, activation_bit=8, weight_percentile=0, act_percentile=0, full_precision_flag=False, quant_act=True),
        # init_cfg=dict(type='Pretrained', checkpoint='/home/niko/BEVFormer/ckpts/resnet50_uniform4/checkpoint.pth.tar'),
        ),
    img_neck=dict(
        type='FPN',
        in_channels=[2048],
        out_channels=_dim_,
        start_level=0,
        add_extra_convs='on_output',
        num_outs=_num_levels_,
        relu_before_extra_convs=True,
        conv_cfg=dict(type='Conv2d', weight_bit=quantize_bit, activation_bit=quantize_bit, full_precision_flag=False, quant_act=False)),
    pts_bbox_head=dict(
        type='BEVFormerHead', 
        bev_h=bev_h_,
        bev_w=bev_w_,
        weight_bit=quantize_bit,
        activation_bit=quantize_bit,#quant_linear_Q
        num_query=900,
        num_classes=10,
        in_channels=_dim_,
        sync_cls_avg_factor=True,
        with_box_refine=True,
        as_two_stage=False,
        transformer=dict(
            type='PerceptionTransformer', 
            rotate_prev_bev=True,
            use_shift=True,
            use_can_bus=True,
            embed_dims=_dim_,
            weight_bit=quantize_bit,
            activation_bit=quantize_bit,#quant_linear_Q
            encoder=dict(
                type='BEVFormerEncoder',
                num_layers=3,
                pc_range=point_cloud_range,
                num_points_in_pillar=4,
                return_intermediate=False,
                transformerlayers=dict(
                    type='BEVFormerLayer',
                    weight_bit=quantize_bit,
                    activation_bit=quantize_bit,  #quant_linear_Q
                    attn_cfgs=[
                        dict(
                            type='TemporalSelfAttention',
                            embed_dims=_dim_,
                            num_levels=1,
                            weight_bit=quantize_bit,
                            activation_bit=quantize_bit,  #quant_linear_Q
                            ),
                        dict(
                            type='SpatialCrossAttention', 
                            pc_range=point_cloud_range,
                            weight_bit=quantize_bit,
                            activation_bit=quantize_bit,#quant_linear_Q
                            deformable_attention=dict(
                                type='MSDeformableAttention3D', 
                                embed_dims=_dim_,
                                num_points=8,
                                num_levels=_num_levels_,
                                weight_bit=quantize_bit,
                                activation_bit=quantize_bit,),#quant_linear_Q
                            embed_dims=_dim_,
                        )
                    ],
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm'))),
            decoder=dict(
                type='DetectionTransformerDecoder',
                num_layers=6,
                return_intermediate=True,
                transformerlayers=dict(
                    type='DetrTransformerDecoderLayer',
                    attn_cfgs=[
                        dict(
                            type='MultiheadAttention',
                            embed_dims=_dim_,
                            num_heads=8,
                            dropout=0.1),
                         dict(
                            type='CustomMSDeformableAttention', 
                            embed_dims=_dim_,
                            num_levels=1,
                            weight_bit=quantize_bit,
                            activation_bit=quantize_bit, #quant_linear_Q
                            ),
                    ],
                    weight_bit=quantize_bit,
                    activation_bit=quantize_bit, #quant_linear_Q
                    feedforward_channels=_ffn_dim_,
                    ffn_dropout=0.1,
                    operation_order=('self_attn', 'norm', 'cross_attn', 'norm',
                                     'ffn', 'norm')))),
        bbox_coder=dict(
            type='NMSFreeCoder',
            post_center_range=[-61.2, -61.2, -10.0, 61.2, 61.2, 10.0],
            pc_range=point_cloud_range,
            max_num=300,
            voxel_size=voxel_size,
            num_classes=10),
        positional_encoding=dict(
            type='LearnedPositionalEncoding',
            num_feats=_pos_dim_,
            row_num_embed=bev_h_,
            col_num_embed=bev_w_,
            ),
        loss_cls=dict(
            type='FocalLoss',
            use_sigmoid=True,
            gamma=2.0,
            alpha=0.25,
            loss_weight=2.0),
        loss_bbox=dict(type='L1Loss', loss_weight=0.25),
        loss_iou=dict(type='GIoULoss', loss_weight=0.0)),
    # model training and testing settings
    train_cfg=dict(pts=dict(
        grid_size=[512, 512, 1],
        voxel_size=voxel_size,
        point_cloud_range=point_cloud_range,
        out_size_factor=4,
        assigner=dict(
            type='HungarianAssigner3D',
            cls_cost=dict(type='FocalLossCost', weight=2.0),
            reg_cost=dict(type='BBox3DL1Cost', weight=0.25),
            iou_cost=dict(type='IoUCost', weight=0.0), # Fake cost. This is just to make it compatible with DETR head.
            pc_range=point_cloud_range))))

# algorithm settings
algorithm = dict(
    type='GeneralDistill',
    architecture=dict(
        type='MMDetArchitecture',
        model=student,
    ),
    distiller=dict(
        type='SingleTeacherDistiller',
        teacher=teacher,
        teacher_trainable=False,
        components=[
            dict(
                student_module='pts_bbox_head.transformer',
                teacher_module='pts_bbox_head.transformer',
                losses=[
                    dict(
                        type='View_dependent_Divergence',
                        name='loss_cwd_BEVFeature',
                        tau=1,
                        loss_weight=5,
                    )
                ])
        ]),
)



dataset_type = 'CustomNuScenesDataset'
data_root = 'data/nuscenes/'
file_client_args = dict(backend='disk')


train_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='PhotoMetricDistortionMultiViewImage'),
    dict(type='LoadAnnotations3D', with_bbox_3d=True, with_label_3d=True, with_attr_label=False),
    dict(type='ObjectRangeFilter', point_cloud_range=point_cloud_range),
    dict(type='ObjectNameFilter', classes=class_names),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
    dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
    dict(type='PadMultiViewImage', size_divisor=32),
    dict(type='DefaultFormatBundle3D', class_names=class_names),
    dict(type='CustomCollect3D', keys=['gt_bboxes_3d', 'gt_labels_3d', 'img'])
]

test_pipeline = [
    dict(type='LoadMultiViewImageFromFiles', to_float32=True),
    dict(type='NormalizeMultiviewImage', **img_norm_cfg),
   
    dict(
        type='MultiScaleFlipAug3D',
        img_scale=(1600, 900),
        pts_scale_ratio=1,
        flip=False,
        transforms=[
            dict(type='RandomScaleImageMultiViewImage', scales=[0.5]),
            dict(type='PadMultiViewImage', size_divisor=32),
            dict(
                type='DefaultFormatBundle3D',
                class_names=class_names,
                with_label=False),
            dict(type='CustomCollect3D', keys=['img'])
        ])
]

data = dict(
    samples_per_gpu=1,
    workers_per_gpu=4,
    train=dict(
        type=dataset_type,
        data_root=data_root,
        ann_file=data_root + 'nuscenes_infos_temporal_train.pkl',
        pipeline=train_pipeline,
        classes=class_names,
        modality=input_modality,
        test_mode=False,
        use_valid_flag=True,
        bev_size=(bev_h_, bev_w_),
        queue_length=queue_length,
        # we use box_type_3d='LiDAR' in kitti and nuscenes dataset
        # and box_type_3d='Depth' in sunrgbd and scannet dataset.
        box_type_3d='LiDAR'),
    val=dict(type=dataset_type,
             data_root=data_root,
             ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
             pipeline=test_pipeline,  bev_size=(bev_h_, bev_w_),
             classes=class_names, modality=input_modality, samples_per_gpu=1),
    test=dict(type=dataset_type,
              data_root=data_root,
              ann_file=data_root + 'nuscenes_infos_temporal_val.pkl',
              pipeline=test_pipeline, bev_size=(bev_h_, bev_w_),
              classes=class_names, modality=input_modality),
    shuffler_sampler=dict(type='DistributedGroupSampler'),
    nonshuffler_sampler=dict(type='DistributedSampler')
)

optimizer = dict(
    type='AdamW',
    lr=2e-4,
    paramwise_cfg=dict(
        custom_keys={
            'img_backbone': dict(lr_mult=0.1),
        }),
    weight_decay=0.01)

optimizer_config = dict(grad_clip=dict(max_norm=0.1, norm_type=2))
# learning policy
lr_config = dict(
    policy='CosineAnnealing',
    warmup='linear',
    warmup_iters=500,
    warmup_ratio=1.0 / 3,
    min_lr_ratio=1e-3)
total_epochs = 48
evaluation = dict(interval=1, pipeline=test_pipeline)

runner = dict(type='EpochBasedRunner', max_epochs=total_epochs)

log_config = dict(
    interval=50,
    hooks=[
        dict(type='TextLoggerHook'),
        dict(type='TensorboardLoggerHook')
    ])

checkpoint_config = dict(interval=1)

# checkpoint = '/home/niko/BEVFormer/ckpts/bevformer_tiny_epoch_24.pth'