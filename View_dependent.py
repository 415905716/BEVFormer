# Copyright (c) OpenMMLab. All rights reserved.
from prometheus_client import Histogram
import torch
import torch.nn as nn
import torch.nn.functional as F

from ..builder import LOSSES


@LOSSES.register_module()
class View_dependent_Divergence(nn.Module):
    """PyTorch version of `Channel-wise Distillation for Semantic Segmentation.

    <https://arxiv.org/abs/2011.13256>`_.

    Args:
        tau (float): Temperature coefficient. Defaults to 1.0.
        loss_weight (float): Weight of loss. Defaults to 1.0.
    """

    def __init__(
        self,
        tau=1.0,
        loss_weight=1.0,
    ):
        super(View_dependent_Divergence, self).__init__()
        self.tau = tau
        self.loss_weight = loss_weight

    
    def forward(self, preds_S, preds_T):
        """Forward computation.

        Args:
            preds_S (torch.Tensor): The student model prediction with
                shape (N, C, H, W).
            preds_T (torch.Tensor): The teacher model prediction with
                shape (N, C, H, W).

        Return:
            torch.Tensor: The calculated loss value.
        """

        bev_embed_S = preds_S[0] #[2500, 1, 256]
        bev_embed_T = preds_T[0]
        B, N, C = bev_embed_S.shape
        
        bev_mask = preds_S[4] #[6, 1, 2500, 4]
        bev_mask_2d = bev_mask.sum(-1)/4

        img_feature_S = preds_S[5][0]
        img_feature_T = preds_T[5][0]
        Bi, Ni, Ci, Hi, Wi = img_feature_S.shape
        
        logsoftmax = torch.nn.LogSoftmax(dim=1)
        

        softmax_pred_T_img = F.softmax(img_feature_T/ self.tau,dim=1)
        loss_img = torch.sum(softmax_pred_T_img * logsoftmax(img_feature_T / self.tau) - 
                             softmax_pred_T_img * logsoftmax(img_feature_S / self.tau),[2,3,4]) *(self.tau**2)
        loss_img = loss_img.transpose(0,1)
        loss_img = loss_img / (Ci * Hi * Wi)
        
        masked_loss_img = loss_img * bev_mask_2d # [6,2500]
        
        softmax_pred_T = F.softmax(bev_embed_T.view(B,-1)/self.tau, dim=1) # [2500, 256]

        loss_BEV = torch.sum(softmax_pred_T *
                         logsoftmax(bev_embed_T.view(B,-1) / self.tau) -
                         softmax_pred_T *
                         logsoftmax(bev_embed_S.view(B,-1) / self.tau),[1]) * (
                             self.tau**2) #[2500]
        loss_BEV = loss_BEV / (C * N)

        loss_VD = masked_loss_img * loss_BEV

        loss = self.loss_weight * loss_VD 

        loss = torch.sum(loss)

        return loss

"""Forward function for `Detr3DTransformer`.
        Args:
            mlvl_feats (list(Tensor)): Input queries from
                different level. Each element has shape
                [bs, num_cams, embed_dims, h, w].
            bev_queries (Tensor): (bev_h*bev_w, c)
            bev_pos (Tensor): (bs, embed_dims, bev_h, bev_w)
            object_query_embed (Tensor): The query embedding for decoder,
                with shape [num_query, c].
            reg_branches (obj:`nn.ModuleList`): Regression heads for
                feature maps from each decoder layer. Only would
                be passed when `with_box_refine` is True. Default to None.
        Returns:
            tuple[Tensor]: results of decoder containing the following tensor.
                - bev_embed: BEV features
                - inter_states: Outputs from decoder. If
                    return_intermediate_dec is True output has shape \
                      (num_dec_layers, bs, num_query, embed_dims), else has \
                      shape (1, bs, num_query, embed_dims).
                - init_reference_out: The initial value of reference \
                    points, has shape (bs, num_queries, 4).
                - inter_references_out: The internal value of reference \
                    points in decoder, has shape \
                    (num_dec_layers, bs,num_query, embed_dims)
                - enc_outputs_class: The classification score of \
                    proposals generated from \
                    encoder's feature maps, has shape \
                    (batch, h*w, num_classes). \
                    Only would be returned when `as_two_stage` is True, \
                    otherwise None.
                - enc_outputs_coord_unact: The regression results \
                    generated from encoder's feature maps., has shape \
                    (batch, h*w, 4). Only would \
                    be returned when `as_two_stage` is True, \
                    otherwise None.
        """