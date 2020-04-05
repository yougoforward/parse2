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
        self.conv0 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))

        self.conv2 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(nn.Conv2d(304, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        # self.conv4 = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, num_classes, kernel_size=1, padding=0, dilation=1, bias=True))
        self.conv4 = nn.Conv2d(256+512, num_classes, kernel_size=1, padding=0, dilation=1, bias=True)
        self.alpha = nn.Parameter(torch.ones(1))
        # self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
        #                          nn.Conv2d(256, 256, 1, bias=False), InPlaceABNSync(256))
        # self.se = nn.Sequential(
        #                     nn.Conv2d(256, 256, 1, bias=True),
        #                     nn.Sigmoid())

    def forward(self, xt, gp, xm, xl):
        _, _, h, w = xm.size()
        xt = self.conv0(F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True) + self.alpha * xm)
        _, _, th, tw = xl.size()
        xt_fea = self.conv1(xt)
        xt = F.interpolate(xt_fea, size=(th, tw), mode='bilinear', align_corners=True)
        xl = self.conv2(xl)
        x = torch.cat([xt, xl], dim=1)
        x_fea = self.conv3(x)
        # gp = self.gap(x_fea)
        # se = self.se(gp)
        # out = torch.cat([x_fea+se*x_fea, gp.expand_as(x_fea)], dim=1)
        n, c, _, _ = gp.size()
        output = torch.cat([x_fea, gp.expand(n, c, th, tw)], dim=1)
        x_seg = self.conv4(output)
        return x_seg, xt_fea


class AlphaHBDecoder(nn.Module):
    def __init__(self, hbody_cls):
        super(AlphaHBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   SEModule(256, reduction=16) 
                                   )
                                   
        # self.cls_hb = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, hbody_cls, kernel_size=1, padding=0, stride=1, bias=True))
        self.cls_hb = nn.Conv2d(256, hbody_cls, kernel_size=1, padding=0, stride=1, bias=True)
        self.alpha_hb = nn.Parameter(torch.ones(1))
        # self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
        #                          nn.Conv2d(256, 256, 1, bias=False), InPlaceABNSync(256))
        # self.se = nn.Sequential(
        #                     nn.Conv2d(256, 256, 1, bias=True),
        #                     nn.Sigmoid())

    def forward(self, x, gp, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_hb * skip
        output = self.conv1(xfuse)
        # gp = self.gap(output)
        # se = self.se(gp)
        # output = torch.cat([output+se*output, gp.expand_as(output)], dim=1)
        # n, c, _, _ = gp.size()
        # output = torch.cat([output, gp.expand(n, c, h, w)], dim=1)
        output = self.cls_hb(output)
        return output


class AlphaFBDecoder(nn.Module):
    def __init__(self, fbody_cls):
        super(AlphaFBDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   SEModule(256, reduction=16) 
                                   )
                                #    SEModule(256, reduction=16)
        # self.cls_fb = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(256, fbody_cls, kernel_size=1, padding=0, stride=1, bias=True))
        self.cls_fb = nn.Conv2d(256, fbody_cls, kernel_size=1, padding=0, stride=1, bias=True)
        self.alpha_fb = nn.Parameter(torch.ones(1))
        # self.gap = nn.Sequential(nn.AdaptiveAvgPool2d(1),
        #                          nn.Conv2d(256, 256, 1, bias=False), InPlaceABNSync(256))
        # self.se = nn.Sequential(
        #                     nn.Conv2d(256, 256, 1, bias=True),
        #                     nn.Sigmoid())

    def forward(self, x, gp, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_fb * skip
        output = self.conv1(xfuse)
        # gp = self.gap(output)
        # se = self.se(gp)
        # output = torch.cat([output+se*output, gp.expand_as(output)], dim=1)
        # n, c, _, _ = gp.size()
        # output = torch.cat([output, gp.expand(n, c, h, w)], dim=1)
        output = self.cls_fb(output)
        return output

class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        # self.layer5 = MagicModule(2048, 512, 1)
        self.layer5 = ASPPModule(2048, 512)
        self.layer5h = ASPPModule(2048, 512)
        self.layer5f = ASPPModule(2048, 512)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = AlphaHBDecoder(hbody_cls)
        self.layerf = AlphaFBDecoder(fbody_cls)
        
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(256), nn.ReLU(inplace=False),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg, gp = self.layer5(x[-1])
        # segh, gph = self.layer5h(x[-1])
        # segf, gpf = self.layer5f(x[-1])


        x_seg, xt_fea = self.layer6(seg, gp, x[1], x[0])
        alpha_hb = self.layerh(seg, gp, x[1])
        alpha_fb = self.layerf(seg, gp, x[1])

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
