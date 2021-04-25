import faster_rcnn as models
import data_loader
import loss_function
import util

import copy
import pickle

import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

img_rate = 5
save_rate = 10
lr = 0.01
momentum = 0.9

#batch_size = 66 #Uses too much VRAM on my 1050ti unfortunately :(
batch_size = 11

def train_model(model, dataloaders, criterion, only_rpn=False, save_dir=None, num_epochs=1000, show_output=False):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    val_loss_history = []
    tr_loss_history = []
    val_iou_history = []
    tr_iou_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_val_loss = 1e6
    best_iou = 1e6

    for epoch in range(num_epochs):
        epoch_train_start = True
        epoch_val_start = True
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode (uses too much VRAM right now for some reason)

            avg_loss = 0
            avg_cls_loss = 0
            avg_reg_loss = 0
            avg_roi_iou = 0
            loop_iters = 0

            for img, inputs, labels, lens in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                lens = lens.to(device)

                if only_rpn:
                    if phase == 'validation':
                        with torch.no_grad():
                            reg, score, anchors, rois = model(inputs)
                    else:
                        reg, score, anchors, rois = model(inputs)

                    epoch_start = (phase == 'training' and epoch_train_start) or (phase == 'validation' and epoch_val_start)
                    if show_output and epoch_start and epoch % img_rate == 0:
                        loss, cls_loss, reg_loss, roi_iou = criterion(reg, score, anchors, labels, lens, rois, device, img=img)
                    else:
                        loss, cls_loss, reg_loss, roi_iou = criterion(reg, score, anchors, labels, lens, rois, device, img=None)
                else:
                    activations = model(inputs)
                    loss = criterion(activations, labels)

                avg_loss += loss
                avg_cls_loss += cls_loss
                avg_reg_loss += reg_loss
                avg_roi_iou += roi_iou
                loop_iters += 1
                if phase == 'training':
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                    epoch_train_start = False
                else:
                    epoch_val_start = False

            #del model.rpn.anchors
            #torch.cuda.empty_cache()
            avg_loss /= loop_iters
            avg_cls_loss /= loop_iters
            avg_reg_loss /= loop_iters
            avg_roi_iou /= loop_iters
            print("Phase: {}, Avg Class Loss:   {}".format(phase, avg_cls_loss))
            print("Phase: {}, Avg Reg Loss:     {}".format(phase, avg_reg_loss))
            print("Phase: {}, Avg Overall Loss: {}".format(phase, avg_loss))
            print("Phase: {}, Avg ROI IOU:      {}".format(phase, avg_roi_iou))

            if phase == 'validation' and avg_loss < best_val_loss:
                best_val_loss = avg_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                if save_dir:
                    torch.save(best_model_wts, os.path.join(save_dir,'RCNN.pth'))
            elif save_dir and phase == 'training' and epoch % save_rate == 0:
                torch.save(model.state_dict(), os.path.join(save_dir,'RCNN_train.pth'))

            if phase == 'validation' and avg_roi_iou < best_iou:
                best_iou = avg_roi_iou

            if phase == 'validation':
                val_loss_history.append(avg_loss)
                val_iou_history.append(avg_roi_iou)
            else:
                tr_loss_history.append(avg_loss)
                tr_iou_history.append(avg_roi_iou)

    print('Best loss: {:4f}'.format(best_val_loss))
    print('Best iou history: {:4f}'.format(best_iou))
    pickle.dump(tr_loss_history, open('tr_his.pkl', 'wb'))
    pickle.dump(val_loss_history, open('val_his.pkl', 'wb'))
    pickle.dump(tr_iou_history, open('tr_iou_his.pkl', 'wb'))
    pickle.dump(val_iou_history, open('val_iou_his.pkl', 'wb'))

def train_rpn(device):
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)

    frcnn = models.Faster_RCNN(device, only_rpn=True).to(device)
    dataloaders = data_loader.get_dataloaders(batch_size)
    train_model(frcnn, dataloaders, loss_function.rpn_loss, save_dir=save_dir, only_rpn=True, show_output=True)

if __name__ == "__main__":
    device = util.get_device()
    train_rpn(device)