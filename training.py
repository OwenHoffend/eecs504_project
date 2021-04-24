import faster_rcnn as models
import data_loader
import loss_function
import util

import torch
import torch.optim as optim
from tqdm import tqdm

lr = 0.01
momentum = 0.9

#batch_size = 66 #Uses too much VRAM on my 1050ti unfortunately :(
batch_size = 6

def train_model(model, dataloaders, criterion, only_rpn=False, save_dir = None, num_epochs=25):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
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
            for img, inputs, labels, lens in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                lens = lens.to(device)

                if only_rpn:
                    reg, score, anchors = model(inputs)
                    loss = criterion(reg, score, anchors, labels, lens, device, img=None)
                else:
                    activations = model(inputs)
                    loss = criterion(activations, labels)

                avg_loss += loss
                loop_iters += 1
                if phase == 'training':
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()
            del model.rpn.anchors
            torch.cuda.empty_cache()
            avg_loss /= loop_iters
            print("Phase: {}, Avg Loss: {}".format(phase, avg_loss))



def train_rpn(device):
    frcnn = models.Faster_RCNN(device, only_rpn=True).to(device)
    dataloaders = data_loader.get_dataloaders(batch_size)
    train_model(frcnn, dataloaders, loss_function.rpn_loss, only_rpn=True)

if __name__ == "__main__":
    device = util.get_device()
    train_rpn(device)

