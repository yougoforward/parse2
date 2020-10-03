import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import ASPPModule

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
class DecoderModule2(nn.Module):
    
    def __init__(self, num_classes):
        super(DecoderModule2, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(256, 256, 1, bias=False),
                            nn.ReLU(True),
                            nn.Conv2d(256, 256, 1, bias=True),
                            nn.Sigmoid())
        self.pred_conv = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True))
        
        self.decoder = DecoderModule3(256,256,256,256)
        
    def forward(self, x, xl):
        out = self.conv0(x)
        out = out + self.se(out)*out
        # refine 
        out = self.decoder(out, xl)
        # prediction
        out = self.pred_conv(out)
        return out

class DecoderModule3(nn.Module):
    
    def __init__(self, in_dim1=256, in_dim2=256, inter_dim=256, out_dim=256):
        super(DecoderModule3, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(in_dim1, 48, kernel_size=1, padding=0, dilation=1, bias=False), BatchNorm2d(48), nn.ReLU(inplace=False))
        # self.conv2 = nn.Sequential(
        #     nn.Conv2d(in_dim2, inter_dim, kernel_size=1, padding=0, dilation=1, bias=False), BatchNorm2d(inter_dim), nn.ReLU(inplace=False))

        self.refine = nn.Sequential(
            nn.Conv2d(in_dim2+48, inter_dim, kernel_size=1, padding=0, dilation=1, bias=False), BatchNorm2d(inter_dim), nn.ReLU(inplace=False),
            nn.Conv2d(inter_dim, out_dim, kernel_size=1, padding=0, dilation=1, bias=False),
            BatchNorm2d(out_dim),nn.ReLU(inplace=False))
        # self.relu = nn.ReLU(inplace=False)
        
    def forward(self, x, xl):
        _,_,h,w = xl.size()
        x = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xlp = self.conv1(xl)
        # xp = self.conv2(x)
        cat = torch.cat([x,xlp], dim=1)
        out = self.refine(cat)
        # out = self.relu(x+out)
        return out
    
class DecoderModule(nn.Module):
    
    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        self.se = nn.Sequential(nn.AdaptiveAvgPool2d(1),
                            nn.Conv2d(256, 256, 1, bias=False),
                            nn.ReLU(True),
                            nn.Conv2d(256, 256, 1, bias=True),
                            nn.Sigmoid())
        self.pred_conv = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True))
    def forward(self, x):
        out = self.conv0(x)
        out = out + self.se(out)*out
        out = self.pred_conv(out)
        return out
class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        self.layer5 = ASPPModule(2048, 512)
        # self.layer_part = DecoderModule(num_classes)
        self.layer_part = DecoderModule2(num_classes)
        self.layer_half = DecoderModule(hbody_cls)
        self.layer_full = DecoderModule(fbody_cls)
        
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(256), nn.ReLU(inplace=False),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.skip = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False),
                                   )
        self.fuse = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False),
                                   nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))
        self.project = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))
    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        _,_,h,w = x[1].size()
        context0 = self.layer5(x[-1])
        context0 = F.interpolate(context0, size=(h, w), mode='bilinear', align_corners=True)
        context1 = self.fuse(torch.cat([self.skip(x[1]), context0], dim=1))
        context = self.project(torch.cat([context0, context1], dim=1))

        # p_seg = self.layer_part(context)
        p_seg = self.layer_part(context, x[0])
        h_seg = self.layer_half(context)
        f_seg = self.layer_full(context)

        return p_seg, h_seg, f_seg, x_dsn


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
