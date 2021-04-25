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

lr = 0.05
momentum = 0.9

#batch_size = 66 #Uses too much VRAM on my 1050ti unfortunately :(
batch_size = 33

def train_model(model, dataloaders, criterion, only_rpn=False, save_dir=None, num_epochs=40, show_output=False):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    val_loss_history = []
    tr_loss_history = []

    best_model_wts = copy.deepcopy(model.state_dict())

    for epoch in range(num_epochs):
        epoch_start = True
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode (uses too much VRAM right now for some reason)

            avg_loss = 0
            loop_iters = 0
            best_loss = 1e6

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

                    if show_output and epoch_start:# and epoch == num_epochs - 1:
                        loss = criterion(reg, score, anchors, labels, lens, rois, device, img=img)

                    else:
                        loss = criterion(reg, score, anchors, labels, lens, rois, device, img=None)
                else:
                    activations = model(inputs)
                    loss = criterion(activations, labels)

                avg_loss += loss
                loop_iters += 1
                if phase == 'training':
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
            
                epoch_start = False

            #del model.rpn.anchors
            #torch.cuda.empty_cache()
            avg_loss /= loop_iters
            print("Phase: {}, Avg Loss: {}".format(phase, avg_loss))

            if phase == 'validation' and avg_loss < best_loss:
                best_loss = avg_loss
                best_model_wts = copy.deepcopy(model.state_dict())

                if save_dir:
                    torch.save(best_model_wts, os.path.join(save_dir,'RCNN.pth'))

            if phase == 'validation':
                val_loss_history.append(avg_loss)
            else:
                tr_loss_history.append(avg_loss)

    print('Best loss: {:4f}'.format(best_loss))
    pickle.dump(tr_loss_history, open('tr_his.pkl', 'wb'))
    pickle.dump(val_loss_history, open('val_his.pkl', 'wb'))

def train_rpn(device):
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)

    frcnn = models.Faster_RCNN(device, only_rpn=True).to(device)
    dataloaders = data_loader.get_dataloaders(batch_size)
    train_model(frcnn, dataloaders, loss_function.rpn_loss, save_dir=save_dir, only_rpn=True, show_output=True)

if __name__ == "__main__":
    device = util.get_device()
    train_rpn(device)