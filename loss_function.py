import torch
import torch.nn.functional as F
import numpy as np
import cv2
import matplotlib.pyplot as plt
import util

def rpn_loss(reg, score, anchors, labels, lens, device, img=None):
    #Intersection over union criteria
    anchors_minmax = util.box_minmax_form(anchors)
    iou = util.batch_jaccard(anchors_minmax, labels)
    N, A, B = iou.shape

    #Thresholding for overlapping anchors
    #These are constant so no need to worry about backprop
    p_star = torch.where(torch.any(iou > 0.5, 2), torch.ones(N,A).to(device), torch.zeros(N,A).to(device))
    p_inds = torch.argmax(iou, axis=1)
    for i in range(N): #Might need to rewrite if these loops are too slow
        for j in range(B):
            ind = p_inds[i, j]
            if iou[i, ind, j] > 0:
                p_star[i, ind] = 1 #Set inds that are maximum IoU

    #p_star : shape [N, A]

    #Display some of the selected anchors just to make sure that this works
    if img != None:
        img_ind = 10
        np_img = np.array(img[img_ind,:,:,:].permute(1,2,0)).astype(np.uint8).copy()
        selected_anchors = p_star[img_ind] == 1
        disp_anchors = anchors_minmax[img_ind, selected_anchors, :]
        util.add_bbs(np_img, disp_anchors, (0,0,255))
        util.add_bbs(np_img, labels[img_ind,0:lens[img_ind],:], (255,0,0))

        plt.imshow(np_img)
        plt.show()

    #Get the ground truth regression parameters
    labels_yxhw = util.box_yxhw_form(labels)
    t_star = torch.zeros_like(reg)

    #For each anchor: compute the regression parameter based on the closest ground truth box
    #Again: these are just constant so no need to worry about backprop
    r_inds = torch.argmax(iou, axis=2) #Should be dim [N, A]

    #TODO: Finish

    #Class loss
    cls_loss = -torch.sum(p_star * torch.log(score + 1e-15)) / (A*N)

    #Compute the overall cost function
    #TODO: Finish

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