import faster_rcnn as models
import data_loader
import loss_function
import util
import copy
import torch
import torch.nn as nn
import torch.optim as optim
from tqdm import tqdm
import os

lr = 0.01
momentum = 0.9

#batch_size = 66 #Uses too much VRAM on my 1050ti unfortunately :(
batch_size = 6

def train_model(model, dataloaders, criterion, only_rpn=False, save_dir = None, num_epochs=25):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)

    val_acc_history = []
    tr_acc_history = []

    best_model_wts = copy.deepcopy(model.state_dict())
    best_acc = 0.0

    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training']: #, 'validation']:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode (uses too much VRAM right now for some reason)

            avg_loss = 0
            loop_iters = 0
            running_loss = 0.0
            running_corrects = 0

            for img, inputs, labels, lens in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                lens = lens.to(device)

                if only_rpn:
                    reg, score, anchors = model(inputs)
                    #preds = #TODO
                    loss = criterion(reg, score, anchors, labels, lens, device, img=None)
                else:
                    activations = model(inputs)
                    #preds = #TODO
                    loss = criterion(activations, labels)

                avg_loss += loss
                loop_iters += 1
                if phase == 'training':
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
            
            
                running_loss += loss.item() * inputs.size(0)
                running_corrects += torch.sum(preds == labels.data)

            del model.rpn.anchors
            torch.cuda.empty_cache()
            avg_loss /= loop_iters
            print("Phase: {}, Avg Loss: {}".format(phase, avg_loss))

            epoch_loss = running_loss / len(dataloaders[phase].dataset)
            epoch_acc = running_corrects.double() / len(dataloaders[phase].dataset)
            print('{} Loss: {:.4f} Acc: {:.4f}'.format(phase, epoch_loss, epoch_acc))
            
            if phase == 'validation' and epoch_acc > best_acc:
                best_acc = epoch_acc
                best_model_wts = copy.deepcopy(model.state_dict())

                if save_dir:
                    torch.save(best_model_wts, os.path.join(save_dir,'RCNN.pth'))

            if phase == 'validation':
                val_acc_history.append(epoch_acc)
            else:
                tr_acc_history.append(epoch_acc)

    print('Best val Acc: {:4f}'.format(best_acc))
    pickle.dump(tr_acc_history, open('tr_his.pkl', 'wb'))
    pickle.dump(val_acc_history, open('val_his.pkl', 'wb'))

def train_rpn(device):
    save_dir = "weights"
    os.makedirs(save_dir, exist_ok=True)

    frcnn = models.Faster_RCNN(device, only_rpn=True).to(device)
    dataloaders = data_loader.get_dataloaders(batch_size)
    train_model(frcnn, dataloaders, loss_function.rpn_loss, save_dir=save_dir, only_rpn=True)

if __name__ == "__main__":
    device = util.get_device()
    train_rpn(device)
    

