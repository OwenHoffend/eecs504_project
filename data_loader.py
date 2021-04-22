import os
import json
import torch
from torchvision import datasets, transforms
from torch.utils.data import Dataset, DataLoader
from PIL import Image

#General info:
#Total number of images: 
    #Train: 1188 (batch size 66 is pretty good)
    #Validation: 399
    #Test: 398
#Total number of labels:
    #Train: 1188
    #Validation: 399
    #Test: Missing 

img_dir = os.path.join(os.getcwd(), "DrivingDataSubsetResized\\images\\")
label_dir = os.path.join(os.getcwd(), "DrivingDataSubsetResized\\labels\\")
max_labels = {'validation':48, 'testing':50, 'training':38}

def get_max_labels():
    for phase in ['training', 'validation']:
        label_loc = os.path.join(label_dir, phase)
        max_labels = 0
        for fn in os.listdir(label_loc):
            with open(os.path.join(label_loc, fn), 'r') as f:
                l = len(json.load(f)['labels'])
                if l > max_labels:
                    max_labels = l
        print("{}: {}".format(phase, max_labels))

class ImgDataSet(Dataset):
    def __init__(self, main_dir, label_dir, transform):
        self.main_dir = main_dir
        self.label_dir = label_dir
        self.transform = transform
        self.all_imgs = os.listdir(main_dir)

    def _get_labels(self, img_name):
        label_loc = os.path.join(self.label_dir, img_name.split('.')[0])
        with open(label_loc + '.json', 'r') as f:
            return json.load(f)['labels']

    def __len__(self):
        return len(self.all_imgs)

    def __getitem__(self, idx):
        img_loc = os.path.join(self.main_dir, self.all_imgs[idx])
        image = Image.open(img_loc).convert("RGB")
        tensor_image = self.transform(image)
        labels = self._get_labels(self.all_imgs[idx])
        boxes = torch.zeros((4, max_labels[self.main_dir.split('\\')[-1]]))
        for idx, box in enumerate(labels):
            box2d = box['box2d']
            boxes[0, idx] = box2d['x1']
            boxes[1, idx] = box2d['x2']
            boxes[2, idx] = box2d['y1']
            boxes[3, idx] = box2d['y2']
        return tensor_image, boxes, len(labels)

def get_dataloaders(batch_size, shuffle = True):

    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    composed_transform = transforms.Compose([
        #transforms.Resize(input_size),
        transforms.ToTensor(),
        transforms.Normalize(mean, std)
    ])
    data_transforms = {
        'training': composed_transform,
        'validation': composed_transform,
        'testing': composed_transform
    }
    image_datasets = {x: ImgDataSet(os.path.join(img_dir, x), os.path.join(label_dir, x), data_transforms[x]) for x in data_transforms.keys()}
    return {x: DataLoader(image_datasets[x], batch_size=batch_size, shuffle=False if x != 'train' else shuffle, num_workers=2) for x in data_transforms.keys()}

if __name__ == "__main__":
    batch_size = 66
    dataloaders_dict = get_dataloaders(batch_size)
    #get_max_labels()
    print('# of training samples {}'.format(len(dataloaders_dict['training'].dataset))) 
    print('# of validation samples {}'.format(len(dataloaders_dict['validation'].dataset)))  
    print('# of test samples {}'.format(len(dataloaders_dict['testing'].dataset)))

    for img, boxes, lens in dataloaders_dict['training']:
        pass