import torch.nn as nn
import torch.nn.functional as F
import torch
from backbone import resnet18
from mrpnet import MRPN
import torch.optim as optim

class HITMRPN(nn.Module):
    def __init__(self, template_size, mode):
        super(HITMRPN, self).__init__()

        self.t_size = template_size
        self.mode = mode
        self.backbone = resnet18()
        self.mrpn1 = MRPN(256)
        self.mrpn2 = MRPN(256)
        self.mrpn3 = MRPN(256)

        self.merge1 = nn.Conv2d(6, 2, kernel_size=1)
        self.merge2 = nn.Conv2d(6, 2, kernel_size=1)

    def __map2point(self, input):
        alpha = 1000.0
        b, _, h, w = input.shape
        f = input[:, 1:2, :, :]
        soft_max = F.softmax(f.view(b, 1, -1)*alpha, dim=2)
        soft_max = soft_max.view(b, -1, h, w)

        indices_kernel = torch.arange(start=0,end=h*w).unsqueeze(0).to(input.device)
        indices_kernel = indices_kernel.view((h, w))

        conv = soft_max*indices_kernel
        indices = conv.sum(2).sum(2)
        
        y = indices%w
        x = (indices/w).floor()%h
        coords = torch.stack([x,y],dim=2)
        return coords.squeeze()

    def __point2value(self, point, feature):
        b = feature.shape[0]
        value = torch.zeros([b, 2]).to(feature.device)
        for i in range(b):
            value[i] = feature[i, :, int(point[i][0]), int(point[i][1])]
        
        return value

    def forward(self, s, t):
        s1, s2, s3 = self.backbone(s)
        t1, t2, t3 = self.backbone(t)

        reg1, cls1 = self.mrpn1(s1, t1)

        reg2, cls2 = self.mrpn2(s2, t2)
        reg2 = F.interpolate(reg2, size=(reg1.shape[2], reg1.shape[3]))
        cls2 = F.interpolate(cls2, size=(reg1.shape[2], reg1.shape[3]))

        reg3, cls3 = self.mrpn3(s3, t3)
        reg3 = F.interpolate(reg3, size=(reg1.shape[2], reg1.shape[3]))
        cls3 = F.interpolate(cls3, size=(reg1.shape[2], reg1.shape[3]))

        reg = torch.cat((reg1, reg2, reg3), dim=1)
        reg = self.merge1(reg)

        cls = torch.cat((cls1, cls2, cls3), dim=1)
        cls = self.merge2(cls)

        anchor_coords = self.__map2point(cls)

        reg = self.__point2value(anchor_coords, reg)
        t_size_tensor = torch.tensor([self.t_size[0], self.t_size[1]]).expand(s.shape[0], 2).to(s.device)
        reg = anchor_coords + reg * t_size_tensor

        if self.mode == 'train':
            cls = self.__point2value(anchor_coords, cls)
            return reg, cls
        
        return reg


if __name__ == "__main__":

    a = torch.rand([2, 3, 100, 100]).cuda()
    b = torch.rand([2, 3, 25, 25]).cuda()
    coord_gt, label_gt = torch.tensor([[2, 10],[3, 9]], dtype=torch.float).cuda(), torch.tensor([1,1]).cuda()
    net = HITMRPN([25, 25]).cuda()
    opt = optim.Adam(net.parameters(), lr=0.02)
    for epoch in range(100):

        coord_pred, label_pred = net(a, b)
        loss1 = 0.1 * F.smooth_l1_loss(coord_pred, coord_gt)
        loss2 = F.cross_entropy(label_pred, label_gt)
        loss = loss1 + loss2
        print(loss1, loss2, loss)
        opt.zero_grad()

        loss.backward()
        opt.step()
        # print(net.backbone.conv1x1_1.state_dict()['weight'][0][0])
        

        