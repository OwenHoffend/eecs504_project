import torch
import torch.nn as nn
import torch.nn.functional as F

import numpy as np
import cv2 #For displaying output only
import matplotlib.pyplot as plt

from collections import OrderedDict

import util

#Network architecture parameters
conv_net_arch = [ #Architecture loosely modeled after AlexNet, with a few changes to make it smaller
    {'out_channels': 48, 'kernel': 9, 'stride': 1, 'padding': 4},
    {'pool': True, 'kernel': 2, 'stride': 2},
    {'out_channels': 128, 'kernel': 5, 'stride': 1, 'padding': 2},
    #{'pool': True, 'kernel': 2, 'stride': 2},
    {'out_channels': 192, 'kernel': 3, 'stride': 1, 'padding': 1},
    {'out_channels': 128, 'kernel': 3, 'stride': 1, 'padding': 1}
]
conv_net_scale_factor = 2 #Divide img W/H by scale factor to get feature map size

rpn_hidden_layers = 64
rpn_a_scales = [8, 16, 30]
#rpn_a_scales = [16, 32, 60]
rpn_a_ratios = [0.5, 1., 2.]
fc_net_arch = [
    #TODO: fill out
]

#Overall top level module
class Faster_RCNN(nn.Module):
    def __init__(self, device, only_rpn=False):
        super(Faster_RCNN, self).__init__()
        self.device = device
        self.only_rpn = only_rpn
        self.cnn = ConvNet(conv_net_arch)
        self.rpn = RPN(device, conv_net_arch[-1]['out_channels'], rpn_hidden_layers, rpn_a_scales, rpn_a_ratios)

        #Rest of the network
        #if not only_rpn:
        #    self.roi_pool = ROI_Pool()
        #    self.fc = FC_Net()

    def forward(self, x):
        feature_map = self.cnn(x)
        #plt.plot(feature_map[0, 0, :, :].to(cpu))
        #plt.show()
        reg, score, rois = self.rpn(feature_map)
        if self.only_rpn:
            return reg, score, self.rpn.anchors * conv_net_scale_factor, rois * conv_net_scale_factor
        else:
            pass

class ConvNet(nn.Module):
    def __init__(self, cfg, in_channels=3, init_weights=True):
        super(ConvNet, self).__init__()
        layers = []
        last_ch = in_channels
        for iter, item in enumerate(cfg):
            ksz = item['kernel']
            stride = item['stride']
            if 'pool' in item.keys():
                layers.append(('mpool' + str(iter), nn.MaxPool2d(kernel_size=ksz, stride=stride, padding=0, dilation=1, ceil_mode=False)))
            else:
                padding = item['padding']
                out_channels = item['out_channels']
                layers.append(('conv' + str(iter), nn.Conv2d(last_ch, out_channels, kernel_size=ksz, stride=stride, padding=padding)))
                layers.append(('relu' + str(iter), nn.ReLU(inplace=True)))
                last_ch = out_channels

        self.net =  nn.Sequential(OrderedDict(layers))

        if init_weights:
            self._initialize_weights()

    def forward(self,x):
        return self.net(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')
    
class RPN(nn.Module):
    def __init__(self, device, in_channels, hl_size, a_scales, a_ratios, conv_size=3, stride=1, padding=1, init_weights=True):
        super(RPN, self).__init__()
        self.device = device
        self.in_channels = in_channels
        self.hl_size = hl_size #Hidden layer size
        self.a_scales = a_scales
        self.a_ratios = a_ratios
        self.k = len(a_scales) * len(a_ratios)
        self.anchors_ = None #Populated by forward (underlying structure)
        self.anchors = None #View of internal anchor data expanded to the correct shape

        self.intermediete = nn.Conv2d(self.in_channels, self.hl_size, kernel_size=conv_size, stride=stride, padding=padding)
        self.relu_inter = nn.ReLU(inplace=True)
        self.classification = nn.Conv2d(self.hl_size, self.k, kernel_size=1)
        self.sigmoid_cls = nn.Sigmoid()
        self.regression = nn.Conv2d(self.hl_size, 4 * self.k, kernel_size=1)

        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        """
        x: Shape (N, C, H, W)
        score: Shape (N, k, H, W)
        reg: Shape (4 * k, H, W)
        """
        N, _, H, W = x.shape
        num_anchors = self.k * H * W
        x = self.relu_inter(self.intermediete(x))
        score = self.sigmoid_cls(self.classification(x)).view(N, num_anchors)

        reg = self.regression(x).view(N, 4, num_anchors).transpose(1,2)

        #Get anchors
        if self.anchors_ == None: #Allow multiple invocations of forward to share anchors (for performance)
            self.anchors_ = torch.zeros((4, self.k, H, W)).to(self.device)

            #Center coordinates correspond to the current index only
            for y in range(H):
                for x in range(W):
                    self.anchors_[0,:,y,x] = y
                    self.anchors_[1,:,y,x] = x

            #Uncomment commented sections below to show anchor perceptive fields
            #image = np.zeros((H,W,3), np.uint8)
            #tcx, tcy = W/2, H/2

            for i, scale in enumerate(self.a_scales):
                for j, ratio in enumerate(self.a_ratios):
                    ah = scale * np.sqrt(ratio)
                    aw = scale * np.sqrt(1./ratio)
                    self.anchors_[2,i*len(self.a_ratios)+j,:,:] = ah
                    self.anchors_[3,i*len(self.a_ratios)+j,:,:] = aw
                    #x1 = int(np.round(tcx-aw/2))
                    #x2 = int(np.round(tcx+aw/2))
                    #y1 = int(np.round(tcy-ah/2))
                    #y2 = int(np.round(tcy+ah/2))
                    #cv2.rectangle(image,(x1,y1),(x2,y2),(255,0,0),1) # add rectangle to image
            #plt.imshow(image)
            #plt.show()
        self.anchors = self.anchors_.expand(N, 4, self.k, H, W).view(N, 4, num_anchors).transpose(1,2) #Shape [N, kHW, 4]

        #Adjust anchors based on computed regression parameters
        rois = torch.zeros_like(reg).to(self.device)
        for i in range(4): #Loop over the anchor params: y, x, H, W
            t = reg[:,:,i]
            anchors = self.anchors[:,:,i]
            if i < 2: #Center location adjustment
                wh = self.anchors[:,:,i+2]
                adjusted = t * wh + anchors
            else: #Height/Width adjustment
                adjusted = torch.exp(t) * anchors
            rois[:,:,i] = adjusted
        return reg, score, rois

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight, mode='fan_out', nonlinearity='relu')

class ROI_Pool(nn.Module): #Non-differentiable, see the Faster R-CNN paper section 3.2 for details
    def __init__(self, PH, PW, scale=1):
        super(ROI_Pool, self).__init__()
        self.pool = nn.AdaptiveMaxPool2d((PH, PW))
        self.PH = PH
        self.PW = PW
        
    def forward(self, feature_map, rois):
        """
        feature_map: Shape (N, C, H, W)
        rois: Shape (N, M, 4) --> [y, x, rH, rW], specified as center & height/width
        (where M = HWk)
        output: Shape (N, C, PH, PW)
        """
        N, C, H, W = feature_map.shape
        _, M, _ = rois.shape
        m = torch.zeros((N, C, self.PH, self.PW))

        #FIXME: The rest of this function needs to have the batch dimension added
        #NOTE: The old M dimension was labelled as N

        #Sanity checks
        assert self.PH <= H and self.PW <= W
        assert (H / self.PH).is_integer() and (W / self.PW).is_integer()

        def get_bounds(a, r, lim): #Helper function. Used since the code for x/y directions is the same
            #Split based on even/odd, to ensure the resulting region is always the correct size
            if r & 1:
                ro = ((r-1)/2).int()
                amax = a+ro
            else:
                ro = (r/2).int()
                amax = a+ro
            amin = a-ro

            #Bounds checking. Cut region if it extends beyond the feature map
            if amin < 0: 
                amin = torch.tensor(0).int()
            if amax > lim:
                amax = torch.tensor(lim-1).int()
            return amin, amax

        #FIXME: Don't expect this loop to work until the batch dimension is added
        for r in range(N):
            y, x, rH, rW = torch.round(rois[r, :]).int() #Region center and height/width
            ymin, ymax = get_bounds(y, rH, H)
            xmin, xmax = get_bounds(x, rW, W)
            region = feature_map[:, ymin:ymax, xmin:xmax]
            m[r, :, :, :] = self.pool(region)
        return m

class FC_Net(nn.Module):
    def __init__(self, cfg, in_channels, init_weights=True):
        super(FC, self).__init__()

        layers = []
        last_ch = in_channels
        for iter, item in enumerate(cfg):
            out_features = item['out_features']
            layers.append(('conv' + str(iter), nn.Linear(in_features=last_ch, out_features=out_features, bias=True)))
            layers.append(('relu' + str(iter), nn.ReLU(inplace=True)))
            #Might need dropout
            layers.append(('dropout' + str(iter), nn.Dropout(0.3, inplace=False)))
            last_ch = out_features

        self.net =  nn.Sequential(OrderedDict(layers))

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        return self.net(x)

    def _initialize_weights(self):
        for m in self.modules():
            if isinstance(m, nn.Linear):
                nn.init.normal_(m.weight, 0, 0.01) #Hyperparameters from HW2.2
                nn.init.constant_(m.bias, 0)

if __name__ == "__main__":
    device = util.get_device()
    """
    Test ROI_Pool
    feature_map: (3, 64, 64)
    rois: (2, 4)
    output: (2, 3, 16, 16)
    """
    #test_features = torch.rand((3, 64, 64))
    #test_features = torch.tensor([
    #    [0.1, 0.2, 0.3],
    #    [0.4, 0.5, 0.6],
    #    [0.7, 0.8, 0.9]
    #])
    #test_features = test_features.view(1, 3, 3)
    #rois = torch.tensor([
    #    [0., 0., 1., 1.],
    #    [1., 1., 1., 3.],
    #    [1., 1., 3., 3.]
    #])
    #r_pool = ROI_Pool(1, 1)
    #result = r_pool.forward(test_features, rois)
    #print(result)

    """
    Test RPN
    """
    #test_anchors = torch.tensor([
    #    [1, 1, 1, 3],
    #    [1, 1, 3, 3]
    #])
    #rpn = RPN(3, 16, test_anchors) #in_channels=3, hl_size=16
    #print(rpn) #print network architecture
    #x = torch.rand((3, 64, 64)).view(1, 3, 64, 64)
    #score, rois = rpn.forward(x)

    """Test Faster_RCNN top class"""
    frcnn = Faster_RCNN(device)
    test_img = torch.rand((1, 3, 90, 160))
    frcnn.forward(test_img)
    print(frcnn)
