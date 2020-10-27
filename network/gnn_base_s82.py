import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import ASPPModule2 as ASPPModule

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')

class DecoderModule(nn.Module):
    
    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv1 = nn.Sequential(
            nn.Conv2d(512, 512, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(512), nn.ReLU(inplace=False),
            nn.Conv2d(512, 256, kernel_size=3, padding=1, bias=False),
            BatchNorm2d(256), nn.ReLU(inplace=False))

        self.conv2 = nn.Sequential(
            nn.Conv2d(256, 48, kernel_size=1, stride=1, padding=0, bias=False),
            BatchNorm2d(48), nn.ReLU(inplace=False))

        self.conv3 = nn.Sequential(
            nn.Conv2d(304, 256, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(256), nn.ReLU(inplace=False),
            nn.Conv2d(256, 256, kernel_size=1, padding=0, bias=False),
            BatchNorm2d(256), nn.ReLU(inplace=False))
        
        self.conv4 = nn.Conv2d(256, num_classes, kernel_size=1, padding=0, bias=True)
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, xt, xm, xl):
        _, _, h, w = xm.size()
        xt = self.conv1(F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True) + self.alpha * xm)
        # _, _, th, tw = xl.size()
        # xt = F.interpolate(xt, size=(th, tw), mode='bilinear', align_corners=True)
        # xl = self.conv2(xl)
        # x = torch.cat([xt, xl], dim=1)
        # x_fea = self.conv3(x)
        
        # x_seg = self.conv4(x_fea)
        return xt

class AlphaDecoder(nn.Module):
    def __init__(self, body_cls):
        super(AlphaDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   SEModule(256, reduction=16) 
                                   )
        self.cls_hb = nn.Conv2d(256, body_cls, kernel_size=1, padding=0, stride=1, bias=True)
        self.alpha_hb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_hb * skip
        output = self.conv1(xfuse)
        # output = self.cls_hb(output)
        return output


class GNN_infer(nn.Module):
    def __init__(self, adj_matrix, upper_half_node=[1, 2, 3, 4], lower_half_node=[5, 6], in_dim=256, hidden_dim=64,
                 cls_p=7, cls_h=3, cls_f=2):
        super(GNN_infer, self).__init__()
        self.cls_p = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim

        # node feature transform 
        self.p_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim * cls_p, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim * cls_p), nn.ReLU(inplace=False))
        self.h_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim * cls_h, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim * cls_h), nn.ReLU(inplace=False))
        self.f_conv = nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim * cls_f, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim * cls_f), nn.ReLU(inplace=False))

        # # node supervision
        # self.node_seg_f = nn.ModuleList([nn.Conv2d(hidden_dim, 1, 1)]*cls_f)
        # self.node_seg_h = nn.ModuleList([nn.Conv2d(hidden_dim, 1, 1)]*cls_h)
        # self.node_seg_p = nn.ModuleList([nn.Conv2d(hidden_dim, 1, 1)]*cls_p)
        
        # node supervision
        self.node_seg = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, xp, xh, xf):
        # gnn inference at stride 8
        # feature transform
        f_node_list = list(torch.split(self.f_conv(xf), self.hidden_dim, dim=1))
        h_node_list = list(torch.split(self.h_conv(xh), self.hidden_dim, dim=1))
        p_node_list = list(torch.split(self.p_conv(xp), self.hidden_dim, dim=1))
        
        # node supervision
        # f_seg = torch.cat([self.node_seg_f[i](f_node_list[i]) for i in range(self.cls_f)], dim=1)
        # h_seg = torch.cat([self.node_seg_h[i](h_node_list[i]) for i in range(self.cls_h)], dim=1)
        # p_seg = torch.cat([self.node_seg_p[i](p_node_list[i]) for i in range(self.cls_p)], dim=1)
        f_seg = []
        h_seg = []
        p_seg = []
        f_seg.append(torch.cat([self.node_seg(node) for node in f_node_list], dim=1))
        h_seg.append(torch.cat([self.node_seg(node) for node in h_node_list], dim=1))
        p_seg.append(torch.cat([self.node_seg(node) for node in p_node_list], dim=1))

        return p_seg, h_seg, f_seg, [], [], [
            ], [], [], [], []



class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        self.layer5 = ASPPModule(2048, 512)
        self.layer_part = DecoderModule(num_classes)
        self.layer_half = AlphaDecoder(hbody_cls)
        self.layer_full = AlphaDecoder(fbody_cls)
        
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(256), nn.ReLU(inplace=False),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))
        
        # adjacent matrix for pascal person 
        self.adj_matrix = torch.tensor(
            [[0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0]], requires_grad=False)
        
        # infer with hierarchical person graph
        self.gnn_infer = GNN_infer(adj_matrix=self.adj_matrix, upper_half_node=[1, 2, 3, 4], lower_half_node=[5, 6],
                                   in_dim=256, hidden_dim=32, cls_p=7, cls_h=3, cls_f=2)
        
    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        _,_,h,w = x[1].size()
        context = self.layer5(x[-1])

        p_fea = self.layer_part(context, x[1], x[0])
        h_fea = self.layer_half(context, x[1])
        f_fea = self.layer_full(context, x[1])

        # gnn infer
        p_seg, h_seg, f_seg, decomp_map_f, decomp_map_u, decomp_map_l, comp_map_f, comp_map_u, comp_map_l, \
        Fdep_att_list= self.gnn_infer(p_fea, h_fea, f_fea)

        return p_seg, h_seg, f_seg, decomp_map_f, decomp_map_u, decomp_map_l, comp_map_f, comp_map_u, comp_map_l, \
        Fdep_att_list, x_dsn

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
    # model = OCNet(Bottleneck, [3, 4, 6, 3], num_classes)  # 50
    model = OCNet(Bottleneck, [3, 4, 23, 3], num_classes)  # 101
    return model
