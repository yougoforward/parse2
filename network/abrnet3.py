import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule, ASPPModule

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

class DecoderModule(nn.Module):
    
    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(512+48, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16))

        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)

    def forward(self, xt, xm, xl):
        _, _, th, tw = xl.size()
        xt_up = F.interpolate(xt, size=(th, tw), mode='bilinear', align_corners=True)
        x_skip = self.conv0(xl)
        xt_fea = self.conv1(torch.cat([xt_up, x_skip], dim=1))
        x_seg = self.conv4(xt_fea)
        return x_seg

class AlphaDecoder(nn.Module):
    def __init__(self, numcls):
        super(AlphaDecoder, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(512, 96, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(96), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(512+96, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16))
        self.cls_hb = nn.Conv2d(256, numcls, kernel_size=1, padding=0, stride=1, bias=True)

    def forward(self, x, skip):
        _, _, h, w = skip.size()
        x_skip = self.conv0(skip)
        x_up = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = self.conv1(torch.cat([x_up, x_skip], dim=1))
        output = self.cls_hb(xfuse)
        return output

class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        # self.layer5 = MagicModule(2048, 512, 1)
        self.layer5 = ASPPModule(2048, 512)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = AlphaDecoder(hbody_cls)
        self.layerf = AlphaDecoder(fbody_cls)
        
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(512), nn.ReLU(inplace=False),
                                       nn.Conv2d(512, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])

        x_seg = self.layer6(seg, x[1], x[0])
        alpha_hb = self.layerh(seg, x[1])
        alpha_fb = self.layerf(seg, x[1])

        return [x_seg, alpha_hb, alpha_fb, x_dsn]


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
