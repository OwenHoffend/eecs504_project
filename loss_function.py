import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import util

def rpn_loss(reg, score, anchors, labels, lens, rois, device, img=None):
    #Intersection over union criteria
    anchors_minmax = util.box_minmax_form(anchors)
    rois_minmax = util.box_minmax_form(rois)
    iou = util.batch_jaccard(anchors_minmax, labels)
    N, A, B = iou.shape

    #Thresholding for overlapping anchors
    #These are constant so no need to worry about backprop
    p_star = torch.where(torch.any(iou > 0.7, 2), torch.ones(N,A).to(device), torch.zeros(N,A).to(device)) #Shape [N, A]
    p_inds = torch.argmax(iou, axis=1)
    for i in range(N): #Might need to rewrite if these loops are too slow
        for j in range(B):
            ind = p_inds[i, j]
            if iou[i, ind, j] > 0:
                p_star[i, ind] = 1 #Set inds that are maximum IoU

    #REGRESSION LOSS
    #For each anchor: compute the regression parameter based on the closest ground truth box
    #Again: these are just constant so no need to worry about backprop
    labels_yxhw = util.box_yxhw_form(labels) #Shape [N, B, 4]
    r_inds = torch.argmax(iou, axis=2) #Shape [N, A]
    gt_star = labels_yxhw.view(N * B, 4)[r_inds.view(N * A), :].view(N, A, 4)

    #Get the ground truth regression parameters
    t_star = torch.zeros_like(reg) #Shape [N, A, 4]. Inherits cuda specification from reg
    t_star[:,:,:2] = (gt_star[:,:,:2] - anchors[:,:,:2])/anchors[:,:,2:]
    t_star[:,:,2:] = torch.log((gt_star[:,:,2:] + 1e-6) / anchors[:,:,2:])
    reg_loss = torch.sum(F.smooth_l1_loss(reg, t_star, reduction='none'), dim=2)
    reg_loss = torch.sum(p_star * reg_loss) / (A*N)

    #CLASS LOSS
    pos_loss = p_star*torch.log(score + 1e-6)
    neg_loss = (1-p_star)*torch.log(1-score + 1e-6)
    cls_loss = -torch.sum(pos_loss)/(torch.sum(p_star) + 1e-6) - torch.sum(neg_loss)/(torch.sum(1-p_star) + 1e-6)


    img_ind = 0 #Arbitrary
    top_values, top_indices = torch.topk(score[img_ind], lens[img_ind])

    #Display some of the selected anchors just to make sure that this works
    lambda_ = 100 #(for now)
    if img != None:
        np_img = np.array(img[img_ind,:,:,:].permute(1,2,0)).astype(np.uint8).copy()
        selected_anchors = p_star[img_ind] == 1
        disp_anchors = anchors_minmax[img_ind, selected_anchors, :]
        util.add_bbs(np_img, disp_anchors, (0,0,255))
        util.add_bbs(np_img, labels[img_ind,0:lens[img_ind],:], (255,0,0))

        #Plot top k bounding boxes based on score
        util.add_bbs(np_img, rois_minmax[img_ind, top_indices, :], (0, 255, 64))
        plt.imshow(np_img)
        plt.show(block=False)
        plt.pause(5)
        plt.close()

    #ROI IOU
    del anchors_minmax
    del iou
    del t_star
    del p_star
    roi_iou = torch.sum(util.batch_jaccard(rois_minmax[img_ind, top_indices, :].view(1, lens[img_ind], 4), \
                    labels[img_ind, :, :].view(1, B, 4))) / (lens[img_ind]*B + 1e-6)

    #OVERALL COST FUNCTION
    return cls_loss + lambda_ * reg_loss, cls_loss, reg_loss, roi_iou

if __name__ == "__main__":
    #Test box_minmax_form
    boxes = torch.tensor([
        [[0,0,2,2],
        [1,1,3,3],
        [4,5,1,2]],
        [[0,0,2,2],
        [1,1,3,3],
        [4,5,1,2]]
    ])
    print(boxes.shape)
    result = util.box_minmax_form(boxes)
    print(result.shape)
    print(result)

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