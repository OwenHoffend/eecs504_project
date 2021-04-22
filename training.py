import faster_rcnn as models
import data_loader
import loss_function

import torch
import torch.optim as optim
from tqdm import tqdm

lr = 0.01
momentum = 0.9

batch_size = 66

def train_model(model, dataloaders, criterion, only_rpn=False, save_dir = None, num_epochs=25):
    optimizer = optim.SGD(model.parameters(), lr=lr, momentum=momentum)
    for epoch in range(num_epochs):
        print('Epoch {}/{}'.format(epoch, num_epochs - 1))
        print('-' * 10)

        # Each epoch has a training and validation phase
        for phase in ['training', 'validation']:
            if phase == 'training':
                model.train()  # Set model to training mode
            else:
                model.eval()   # Set model to evaluate mode

            for inputs, labels, lens in tqdm(dataloaders[phase]):
                inputs = inputs.to(device)
                labels = labels.to(device)
                lens = lens.to(device)

                if only_rpn:
                    reg, score = model(inputs)
                    loss = criterion()
                else:
                    activations = model(inputs)
                    loss = criterion(activations, labels)

                if phase == 'training':
                    model.zero_grad()
                    loss.backward()
                    optimizer.step()

def train_rpn():
    frcnn = models.Faster_RCNN(only_rpn=True).to(device)
    dataloaders = data_loader.get_dataloaders(batch_size)
    train_model(frcnn, dataloaders, loss_function.rpn_loss, only_rpn=True)

if __name__ == "__main__":
    device = torch.device("cuda:0" if torch.cuda.is_available() else "cpu")
    if torch.cuda.is_available():
        print("Using the GPU!")
    else:
        print("WARNING: Could not find GPU! Using CPU only.")
    train_rpn()

