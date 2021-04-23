import torch
import torch.nn.functional as F
import numpy as np

def box_minmax_form(boxes):
    """
    Convert from y,x,h,w form to y1,x1,y2,x2 form for a shape [N, M, 4] batch of boxes
    Returns another shape [N, M, 4] tensor
    """
    return torch.cat((boxes[:,:,:2] - boxes[:,:,2:]/2,     # ymin, xmin
                     boxes[:,:,:2] + boxes[:,:,2:]/2), 2)  # ymax, xmax

def box_yxhw_form(boxes):
    """
    Convert from y1,x2,y2,x2 form to y,x,h,w form for a shape [N, M, 4] batch of boxes
    Returns another shape [N, M, 4] tensor
    """
    return torch.cat(((boxes[:,:,2:] + boxes[:,:,:2])/2,  # cx, cy
                     boxes[:,:,2:] - boxes[:,:,:2]), 2)  # w, h

def batch_jaccard(anchors, labels):
    """
    Computes jaccard similarity (IoU) for a batch of anchor boxes
    anchors is of shape: [N, A, 4]
    labels is of shape: [N, B, 4]
    Output is of shape: [N, A, B]
    """
    N, A, _ = anchors.shape
    B = labels.size(1)
    max_xy = torch.min(anchors[:,:,2:].unsqueeze(2).expand(N, A, B, 2),
                        labels[:,:,2:].unsqueeze(1).expand(N, A, B, 2))
    min_xy = torch.max(anchors[:,:,:2].unsqueeze(2).expand(N, A, B, 2),
                        labels[:,:,:2].unsqueeze(1).expand(N, A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    intersect = inter[:,:,:,0] * inter[:,:,:,1]

    area_a = ((anchors[:,:,2]-anchors[:,:,0]) *
              (anchors[:,:,3]-anchors[:,:,1])).unsqueeze(2).expand_as(intersect)
    area_b = ((labels[:,:,2]-labels[:,:,0]) *
              (labels[:,:,3]-labels[:,:,1])).unsqueeze(1).expand_as(intersect)
    union = area_a + area_b - intersect
    return intersect / union

def rpn_loss(reg, score, anchors, labels, device):
    #Intersection over union criteria
    anchors_minmax = box_minmax_form(anchors)
    iou = batch_jaccard(anchors_minmax, labels)
    N, A, B = iou.shape

    #Thresholding for overlapping anchors
    p_star = torch.where(torch.any(iou > 0.7, 2), torch.ones(N,A).to(device), torch.zeros(N,A).to(device))
    p_inds = torch.argmax(iou, axis=1)
    for i in range(N): #Might need to rewrite if these loops are too slow
        for j in range(B):
            ind = p_inds[i, j]
            if ind != 0:
                p_star[i, ind] = 1 #Set inds that are maximum IoU

    #Get the ground truth regression parameters
    labels_yxhw = box_yxhw_form(labels)
    t_star = torch.zeros_like(reg)

    #TODO: Finish

    #Class loss
    cls_loss = -torch.sum(p_star * torch.log(score + 1e-15)) / (A*N)

    #Compute the overall cost function
    #TODO: Finish

if __name__ == "__main__":
    pass
    #Test box_minmax_form
    #boxes = torch.tensor([
    #    [[0,0,2,2],
    #    [1,1,3,3],
    #    [4,5,1,2]],
    #    [[0,0,2,2],
    #    [1,1,3,3],
    #    [4,5,1,2]]
    #])
    #print(boxes.shape)
    #result = box_minmax_form(boxes)
    #print(result.shape)
    #print(result)

    #Test intersect
    #boxes1 = torch.tensor([
    #    [[0, 0, 2, 2], 
    #     [0, 0, 3, 3]],
    #    [[0, 0, 2, 2],
    #     [0, 0, 3, 3]]
    #])

    #boxes2 = torch.tensor([
    #    [[1, 1, 2, 2], 
    #     [0, 0, 3, 3],
    #     [0, 0, 1, 1]],
    #    [[1, 1, 2, 2],
    #     [0, 0, 3, 3],
    #     [1, 1, 1, 1]]
    #])

    #sect = intersect(boxes1, boxes2)
    #print(sect.shape)
    #print(sect)