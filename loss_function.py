import torch
import torch.nn.functional as F
import numpy as np

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

# Fast RCNN loss
def roi_loss(gt_roi_locs, gt_roi_labels, roi_cls_score, roi_cls_loc):
    
    gt_roi_loc = torch.from_numpy(gt_roi_locs)
    gt_roi_label = torch.from_numpy(np.float32(gt_roi_labels)).long()
    
    #Classification loss
    roi_cls_loss = F.cross_entropy(roi_cls_score, gt_roi_label, ignore_index=-1)
    
    #Regression loss
    n_sample = roi_cls_loc.shape[0] 
    roi_loc = roi_cls_loc.view(n_sample, -1, 4) #([128,2,4])
    roi_loc = roi_loc[torch.arange(0, n_sample).long(), gt_roi_label] #([128,4])
    pos = gt_roi_label > 0
    mask = pos.unsqueeze(1).expand_as(roi_loc) #([128,4])
    mask_loc_preds = roi_loc[mask].view(-1,4)
    mask_loc_targets = gt_roi_loc[mask].view(-1,4)
    x = torch.abs(mask_loc_targets - mask_loc_pred)
    roi_loc_loss = ((x<1).float()*0.5*x**2) + ((x>=1).float()*(x-0.5))
    
    roi_lambda = 10
    
    total_loss = roi_cls_loss + (roi_lambda * roi_loc_loss)
    
    return total_loss