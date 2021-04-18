#!/usr/bin/env python
# coding: utf-8

import torch
import torchvision
import torch.nn as nn
import torch.nn.functional as F
from torchvision import transforms
import matplotlib.pyplot as plt
import numpy as np
import cv2
from PIL import Image

# RPN Loss
# pred_anchor_locs :output from the regression layer
# pred_cls_scores :output from the classfication layer
def rpn_loss(pred_anchor_locs, pre_cls_scores, anchor_locations, anchor_labels):
    #Classification
    pred_anchor_locs = pred_anchor_locs.permute(0, 2, 3, 1).contiguous().view(1, -1, 4)
    pred_cls_scores = pre_cls_scores.permute(0, 2, 3, 1).contiguous().view(1, -1 , 2)
    rpn_loc = pred_anchor_locs[0]
    rpn_score = pre_cls_scores[0]
    gt_rpn_loc = torch.from_numpy(anchor_locations)
    gt_rpn_score = torch.from_numpy(anchor_labels)
    rpn_cls_loss = F.cross_entropy(rpn_score, gt_rpn_score_long(), ignore_index = -1)
    #Regression
    pos = gt_rpn_score > 0
    mask = pos.unsqueeze(1).expand_as(rpn_loc) #?
    mask_loc_preds = rpn_loc[mask].view(-1, 4)
    mask_loc_targets = gt_rpn_loc[mask].view(-1, 4)
    
    x = torch.abs(mask_loc_targets - mask_loc_preds)
    rpn_loc_loss = ((x<1).float() * 0.5 * x**2)+((x >= 1).float() * (x-0.5))
    
    rpn_lambda = 10
    N_reg = (gt_rpn_score > 0).float().sum()
    rpn_loc_loss = rpn_cls_loss.sum() / N_reg
    total_loss = rpn_cls_loss + (rpn_lambda * rpn_loc_loss)
    
    return total_loss

