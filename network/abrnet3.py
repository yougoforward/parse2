import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule, ASPPModule2

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

# class DecoderModule(nn.Module):

#     def __init__(self, num_classes):
#         super(DecoderModule, self).__init__()
#         self.conv1 = nn.Sequential(nn.Conv2d(2*512, 256, kernel_size=3, padding=1, bias=False),
#                                    BatchNorm2d(256), nn.ReLU(inplace=False))

#         self.pred_conv = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True))

#     def forward(self, x, gp):
#         n,c,h,w = x.size()
#         out = torch.cat([x,gp.expand_as(x)],dim=1)
#         out = self.conv1(out)
#         out = self.pred_conv(out)
#         return out

class DecoderModule(nn.Module):

    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        self.gp = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        self.pred_conv = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True))

    def forward(self, x, gp):
        x=self.conv0(x)
        out = torch.cat([x,self.gp(gp).expand_as(x)],dim=1)
        out = self.conv1(out)
        out = self.pred_conv(out)
        return out

class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        # self.layer5 = MagicModule(2048, 512, 1)
        self.layer5 = ASPPModule2(2048, 512)
        self.layer_part = DecoderModule(num_classes)
        self.layer_half = DecoderModule(hbody_cls)
        self.layer_full = DecoderModule(fbody_cls)
        
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(256), nn.ReLU(inplace=False),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        # self.project = nn.Sequential(nn.Conv2d(2048, 512, kernel_size=3, padding=1, bias=False),
        #                            BatchNorm2d(512), nn.ReLU(inplace=False))
        self.skip = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))
        self.fuse = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        _,_,h,w = x[1].size()
        context, gp = self.layer5(x[-1])
        # x[-1] = F.interpolate(self.project(x[-1]), size=(h, w), mode='bilinear', align_corners=True)
        # x[-1] = self.fuse(torch.cat([self.skip(x[1]), x[-1]], dim=1))

        context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)
        context = self.fuse(torch.cat([self.skip(x[1]), context], dim=1))

        seg_part = self.layer_part(context, gp)
        seg_half = self.layer_half(context, gp)
        seg_full = self.layer_full(context, gp)

        return [seg_part, seg_half, seg_full, x_dsn]


class OCNet(nn.Module):
    def __init__(self, block, layers, num_classes):
        super(OCNet, self).__init__()
        self.encoder = ResGridNet(block, layers)
        self.decoder = Decoder(num_classes=num_classes)

        for m in self.modules():
            if isinstance(m, nn.Conv2d):
                nn.init.kaiming_normal_(m.weight.data)
            elif isinstance(m, InPlaceABNSync):
                m.weight.data.fill_(1)
                m.bias.data.zero_()

    def forward(self, x):
        x = self.encoder(x)
        x = self.decoder(x)
        return x


def get_model(num_classes=20):
    # model = OCNet(Bottleneck, [3, 4, 6, 3], num_classes) #50
    model = OCNet(Bottleneck, [3, 4, 23, 3], num_classes) #101
    # model = OCNet(Bottleneck, [3, 8, 36, 3], num_classes)  #152
    return model
