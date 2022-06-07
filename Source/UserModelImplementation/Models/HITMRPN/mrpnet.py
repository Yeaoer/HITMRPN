from tkinter.tix import Tree
import torch.nn as nn
import torch
import numpy as np


class SMRPN(nn.Module):
    def __init__(self, inchannels):
        super(SMRPN, self).__init__()

        self.conv3x3_1 = nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1)
        self.bn1 = nn.BatchNorm2d(inchannels)

        self.conv3x3_2 = nn.Conv2d(inchannels, inchannels, kernel_size=3, stride=1, padding=1)
        self.bn2 = nn.BatchNorm2d(inchannels)

        
        self.conv1x1_1 = nn.Conv2d(1, 1, kernel_size=1)
        self.bn3 = nn.BatchNorm2d(1)
        self.relu = nn.ReLU(inplace=True)
        self.conv1x1_2 = nn.Conv2d(1, 2, kernel_size=1)

    def __ncc(self, s, t):
        b, _, s_h, s_w = s.shape
        t_h, t_w = t.shape[2:]
        h, w = s_h - t_h + 1, s_w - t_w + 1
        ncc = torch.zeros((b, 1, h, w)).to(s.device)
        tt = t - torch.mean(t, dim=(1, 2, 3), keepdim=True)
        tt_std = torch.std(t, dim=(1, 2, 3), keepdim=True)
        for i in range(h):
            for j in range(w):
                s_sub = s[:, :, i:i+t_h, j:j+t_w]
                ss = s_sub - torch.mean(s_sub, dim=(1, 2, 3), keepdim=True)
                ss_std = torch.std(s_sub, dim=(1, 2, 3), keepdim=True)
                ncc[:, 0, i, j] = torch.mean(torch.multiply(tt, ss)/(tt_std*ss_std), dim=(1, 2, 3))
        return ncc

    def forward(self, s, t):
        s = self.bn1(self.conv3x3_1(s))
        t = self.bn2(self.conv3x3_2(t))

        ncc = self.__ncc(s, t)

        return self.conv1x1_2(self.relu(self.bn3(self.conv1x1_1(ncc))))


class MRPN(nn.Module):
    def __init__(self, inchannels):
        super(MRPN, self).__init__()

        self.reg = SMRPN(inchannels)
        self.cls = SMRPN(inchannels)

    def forward(self, s, t):
        return self.reg(s, t), self.cls(s, t)


                





if __name__ == "__main__":
    net = MRPN(10)
    a = torch.rand([3, 10, 20, 20])
    b = torch.rand([3, 10, 10, 10])
    c = net(b, a)
    # print(c.shape)
    # c = np.mean(np.multiply((a-np.mean(a)),(b-np.mean(b))))/(np.std(a)*np.std(b))
    print(c[0].shape, c[1].shape)