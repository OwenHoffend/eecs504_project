import torch
import numpy as np
import cv2

def get_device():
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only.")
    return device

def box_minmax_form(boxes):
    """
    Convert from y,x,h,w form to x1,x2,y1,y2 form for a shape [N, M, 4] batch of boxes
    Returns another shape [N, M, 4] tensor
    """
    #mins = boxes[:,:,:2] - boxes[:,:,2:]/2
    #maxes = boxes[:,:,:2] + boxes[:,:,2:]/2
    #return torch.stack((mins[:,:,1], maxes[:,:,1], # x1, x2
    #                 mins[:,:,0], maxes[:,:,0]), 2)  # y1, y2

    return torch.cat((boxes[:,:,:2] - boxes[:,:,2:]/2,     # ymin, xmin
                     boxes[:,:,:2] + boxes[:,:,2:]/2), 2)  # ymax, xmax

def box_yxhw_form(boxes):
    """
    Convert from y1,x2,y2,x2 form to y,x,h,w form for a shape [N, M, 4] batch of boxes
    Returns another shape [N, M, 4] tensor
    """
    return torch.cat(((boxes[:,:,2:] + boxes[:,:,:2])/2,  # cx, cy
                     boxes[:,:,2:] - boxes[:,:,:2]), 2)  # w, h

def jaccard(anchors, labels):
    A, _ = anchors.shape
    B, _ = labels.shape
    max_xy = torch.min(anchors[:,2:].unsqueeze(1).expand(A, B, 2),
                        labels[:,2:].unsqueeze(0).expand(A, B, 2))
    min_xy = torch.max(anchors[:,:2].unsqueeze(1).expand(A, B, 2),
                        labels[:,:2].unsqueeze(0).expand(A, B, 2))
    inter = torch.clamp((max_xy - min_xy), min=0)
    intersect = inter[:,:,0] * inter[:,:,1]

    area_a = ((anchors[:,2]-anchors[:,0]) *
            (anchors[:,3]-anchors[:,1])).unsqueeze(1).expand_as(intersect)
    area_b = ((labels[:,2]-labels[:,0]) *
            (labels[:,3]-labels[:,1])).unsqueeze(0).expand_as(intersect)
    union = area_a + area_b - intersect
    return intersect / union

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

def add_bbs(img, boxes, color):
    for box in boxes:
        y1 = int(torch.round(box[0]))
        x1 = int(torch.round(box[1]))
        y2 = int(torch.round(box[2]))
        x2 = int(torch.round(box[3]))
        cv2.rectangle(img,(x1,y1),(x2,y2),color,1) # add rectangle to image
    return img