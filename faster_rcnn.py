import torch
import torch.nn as nn
import torch.nn.functional as F

from collections import OrderedDict

#Overall top level module
class Faster_RCNN(nn.Module):
    def __init__(self, init_weights=True):
        super(Faster_RCNN, self).__init__()
        #Submodules:
        #CNN: Input --> Feature Map
        self.cnn = ConvNet()
        #RPN
        self.rnp = RPN()
        #ROI Pooling
        self.roi_pool = ROI_Pool() #TODO Needs PH, PW
        #FC Network

        if init_weights:
            self._initialize_weights()

    def forward(self, x):
        pass

    def _initialize_weights(self):
        pass

class ConvNet(nn.Module):
    def __init__(self, cfg, in_channels=3, init_weights=True):
        super(ConvNet, self).__init__()
        layers = []
        last_ch = in_channels
        for iter, item in enumerate(cfg):
            stride = item['stride']
            ksz = item['kernel']
            padding = item['padding']
            out_channels = item['out_channels']
            #Might need pooling
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
    def __init__(self, in_channels, hl_size, anchors, conv_size=3, stride=1, padding=1, init_weights=True):
        super(RPN, self).__init__()
        self.in_channels = in_channels
        self.hl_size = hl_size #Hidden layer size
        self.anchors = anchors #Shape: (k, 4) --> [y, x, aH, aW]
        self.k, _ = anchors.shape

        self.intermediete = nn.Conv2d(self.in_channels, self.hl_size, kernel_size=conv_size, stride=stride, padding=padding)
        self.relu_inter = nn.ReLU(inplace=True)
        self.classification = nn.Conv2d(self.hl_size, self.k, kernel_size=1)
        self.relu_cls = nn.ReLU(inplace=True)
        self.regression = nn.Conv2d(self.hl_size, 4 * self.k, kernel_size=1)
        self.relu_reg = nn.ReLU(inplace=True)

        if init_weights:
            self._initialize_weights()
        
    def forward(self, x):
        """
        x: Shape (C, H, W)
        score: Shape (k, H, W)
        reg: Shape (4 * k, H, W)
        """
        _, _, H, W = x.shape
        x = self.relu_inter(self.intermediete(x))
        score = self.relu_cls(self.classification(x))
        reg = self.relu_reg(self.regression(x).view(4, self.k, H, W)) #Shape [1, 4 * k, H, W] --> [4, k, H, W]

        #Adjust anchors based on computed regression parameters
        rois = torch.zeros_like(reg)
        for i in range(4): #Loop over the anchor params: y, x, H, W
            t = reg[i,:,:,:]
            anchors = self.anchors[:, i].view(self.k, 1, 1)
            if i < 2: #Center location adjustment
                wh = self.anchors[:, i+2].view(self.k, 1, 1)
                adjusted = t * wh + anchors
            else: #Height/Width adjustment
                adjusted = torch.exp(t) * anchors
            #TODO: May need to convert these to integers
            rois[i,:,:,:] = adjusted
        return score, rois.view(4, self.k * H * W).T #rois output is reshaped to conform to ROI_Pool input requirements

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
        feature_map: Shape (C, H, W)
        rois: Shape (N, 4) --> [y, x, rH, rW], specified as center & height/width
        (where N = HWk)
        output: Shape (N, C, PH, PW)
        """
        C, H, W = feature_map.shape
        N, _ = rois.shape
        m = torch.zeros((N, C, self.PH, self.PW))

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

        for r in range(N):
            y, x, rH, rW = torch.round(rois[r, :]).int() #Region center and height/width
            ymin, ymax = get_bounds(y, rH, H)
            xmin, xmax = get_bounds(x, rW, W)
            region = feature_map[:, ymin:ymax, xmin:xmax]
            m[r, :, :, :] = self.pool(region)
        return m

class FC(nn.Module):
    def __init__(self, cfg, init_weights=True):
        super(FC, self).__init__()

        layers = []
        last_ch = in_channels
        for iter, item in enumerate(cfg):
            out_features = item['out_features']
            layers.append(('conv' + str(iter), nn.Linear(in_features=last_ch, out_features=out_features, bias=True)))
            layers.append(('relu' + str(iter), nn.ReLU(inplace=True)))
            #Might need dropout
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
        [0., 0., 1., 1.],
        [1., 1., 1., 3.],
        [1., 1., 3., 3.]
    ])
    r_pool = ROI_Pool(1, 1)
    result = r_pool.forward(test_features, rois)
    print(result)

    """
    Test RPN
    """
    test_anchors = torch.tensor([
        [1, 1, 1, 3],
        [1, 1, 3, 3]
    ])
    rpn = RPN(3, 16, test_anchors) #in_channels=3, hl_size=16
    print(rpn) #print network architecture
    x = torch.rand((3, 64, 64)).view(1, 3, 64, 64)
    score, rois = rpn.forward(x)