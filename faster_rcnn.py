import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

#Overall top level module
class Faster_RCNN(nn.Module):
    def __init__(self):
        super(Faster_RCNN, self).__init__()
        #Submodules:
        #CNN: Input --> Feature Map
        self.cnn = ConvNet()
        #RPN
        self.rnp = RPN()
        #ROI Pooling
        self.roi_pool = ROI_Pool() #TODO Needs PH, PW
        #FC Network

    def forward(self, x):
        pass

class ConvNet(nn.Module):
    def __init__(self, cfg, in_channels=3):
        super(ConvNet, self).__init__()
        layers = []
        last_ch = in_channels
        for iter, item in enumerate(cfg):
            stride = item['stride']
            ksz = item['kernel']
            padding = item['padding']
            out_channels = item['out_channels']
            layers.append(('conv' + str(iter), nn.Conv2d(last_ch, out_channels, kernel_size=ksz, stride=stride, padding=padding)))
            layers.append(('relu' + str(iter), nn.ReLU(inplace=True)))
            last_ch = out_channels

        self.net =  nn.Sequential(OrderedDict(layers))

    def forward(self,x):
        return self.net(x)
    
class RPN(nn.Module):
    def __init__(self, inputSize, layerSize, convSize, anchorNum):
        super(RPN, self).__init__()
        self.inputSize = inputSize
        self.layerSize = layerSize
        self.convSize = convSize
        self.anchorNum = anchorNum

        self.intermediete = nn.Conv2d(self.inputSize, self.layerSize, self.convSize, stride = 1)
        self.relu1 = nn.ReLU(inplace=True)
        self.classificaiton = nn.Conv2D(self.layerSize, self.anchorNum, 1)
        self.regression = nn.Conv2d(self.layerSize, 4 * self.anchorNum, 1)
        
    def forward(self, x):
        """
        x: Shape (C, H, W)
        """
        x = self.relu1(self.intermediete(x))
        score = self.classifation(x)
        reg = self.regression(x)
        return score, reg #<-- reg is still given wrt anchor coordinates

class ROI_Pool(nn.Module):
    def __init__(self, PH, PW, scale=1):
        super(ROI_Pool, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d((PH, PW))
        self.PH = PH
        self.PW = PW
        
    def forward(self, feature_map, rois):
        """
        feature_map: Shape (C, H, W)
        rois: Shape (N, 4) --> [y, x, rH, rW]
        output: Shape (N, C, PH, PW)
        """
        C, H, W = feature_map.shape
        N, _ = rois.shape
        m = torch.zeros((N, C, self.PH, self.PW))

        #Sanity checks
        assert self.PH <= H and self.PW <= W
        assert (H / self.PH).is_integer() and (W / self.PW).is_integer()

        for r in range(N):
            y, x, rH, rW = rois[r, :]
            region = feature_map[:, y:y+rH, x:x+rW]
            m[r, :, :, :] = self.pool(region)
        return m

class FC(nn.Module):
    def __init__(self):
        super(FC, self).__init__()

if __name__ == "__main__":
    """
    Test ROI_Pool
    feature_map: (3, 64, 64)
    rois: (2, 4)
    output: (2, 3, 16, 16)
    """
    #test_features = torch.rand((3, 64, 64))
    test_features = torch.tensor([
        [0.1, 0.2, 0.3],
        [0.4, 0.5, 0.6],
        [0.7, 0.8, 0.9]
    ])
    test_features = test_features.view(1, 3, 3)
    rois = torch.tensor([
        [0, 0, 2, 2],
        [1, 0, 1, 3],
        [1, 1, 2, 2]
    ])
    r_pool = ROI_Pool(2, 2)
    result = r_pool.forward(test_features, rois)
    print(result)