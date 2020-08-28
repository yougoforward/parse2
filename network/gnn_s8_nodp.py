import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule, ASPPModule
from modules.senet import se_resnext50_32x4d, se_resnet101, senet154

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
# from modules.convGRU import ConvGRU
class DecoderModule(nn.Module):

    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))

    def forward(self, x):
        out = self.conv0(x)
        return out

class Full_Graph(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Full_Graph, self).__init__()
        self.cls_f = cls_f
        self.comp_full = nn.Sequential(nn.Conv2d(3*hidden_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
        self.conv_bg = nn.Sequential(nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))

    def forward(self, f_node_list, h_node_list, p_node_list, xf):
        f_node_list_new = []
        for i in range(self.cls_f):
            if i==0:
                node = self.conv_bg(torch.cat([f_node_list[0], h_node_list[0]], dim=1))
            elif i==1:
                node = self.comp_full(torch.cat([f_node_list[1], h_node_list[1], h_node_list[2]], dim=1))
            f_node_list_new.append(node)

        return f_node_list_new

class Half_Graph(nn.Module):
    def __init__(self, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], in_dim=256, hidden_dim=10, cls_p=7,
                 cls_h=3, cls_f=2):
        super(Half_Graph, self).__init__()
        self.cls_h = cls_h
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.upper_parts_len = len(upper_part_list)
        self.lower_parts_len = len(lower_part_list)
        self.decomp_full = nn.Sequential(nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
        self.comp_u = nn.Sequential(nn.Conv2d((self.upper_parts_len+1)*hidden_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
        self.comp_l = nn.Sequential(nn.Conv2d((self.lower_parts_len+1)*hidden_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
        self.conv_bg = nn.Sequential(nn.Conv2d(3*hidden_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))

    def forward(self, f_node_list, h_node_list, p_node_list, xh):
        upper_parts = []
        for part in self.upper_part_list:
            upper_parts.append(p_node_list[part])

        lower_parts = []
        for part in self.lower_part_list:
            lower_parts.append(p_node_list[part])

        h_node_list_new = []
        for i in range(self.cls_h):
            if i==0:
                node = self.conv_bg(torch.cat([f_node_list[0], h_node_list[0],p_node_list[0]], dim=1))
            elif i==1:
                comp = self.comp_u(torch.cat(upper_parts+[h_node_list[1]], dim=1))
                decomp = self.decomp_full(torch.cat([f_node_list[1], h_node_list[1]], dim=1))
                node = comp+decomp
            elif i==2:
                comp = self.comp_l(torch.cat(lower_parts+[h_node_list[2]], dim=1))
                decomp = self.decomp_full(torch.cat([f_node_list[1], h_node_list[2]], dim=1))
                node = comp+decomp
            h_node_list_new.append(node)

        return h_node_list_new


class Part_Graph(nn.Module):
    def __init__(self, adj_matrix, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], in_dim=256, hidden_dim=10,
                 cls_p=7, cls_h=3, cls_f=2):
        super(Part_Graph, self).__init__()
        self.cls_p = cls_p
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list

        self.decomp_half = nn.Sequential(nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
        self.conv_bg = nn.Sequential(nn.Conv2d(2*hidden_dim, hidden_dim, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
    def forward(self, f_node_list, h_node_list, p_node_list, xp):
        p_node_list_new = []
        for i in range(self.cls_p):
            if i==0:
                node = self.conv_bg(torch.cat([h_node_list[0], p_node_list[0]], dim=1))
            elif i in self.upper_part_list:
                node = self.decomp_half(torch.cat([h_node_list[1], p_node_list[i]], dim=1))

            elif i  in self.lower_part_list:
                node = self.decomp_half(torch.cat([h_node_list[2], p_node_list[i]], dim=1))

            p_node_list_new.append(node)
        return p_node_list_new


class GNN(nn.Module):
    def __init__(self, adj_matrix, upper_half_node=[1, 2, 3, 4], lower_half_node=[5, 6], in_dim=256, hidden_dim=10,
                 cls_p=7, cls_h=3, cls_f=2):
        super(GNN, self).__init__()
        self.cp = cls_p
        self.ch = cls_h
        self.cf = cls_f
        self.ch_in = in_dim
        self.hidden = hidden_dim
        self.upper_half_node = upper_half_node
        self.upper_node_len = len(self.upper_half_node)
        self.lower_half_node = lower_half_node
        self.lower_node_len = len(self.lower_half_node)

        self.full_infer = Full_Graph(in_dim, hidden_dim, cls_p, cls_h, cls_f)
        self.half_infer = Half_Graph(self.upper_half_node, self.lower_half_node, in_dim, hidden_dim, cls_p, cls_h,
                                     cls_f)
        self.part_infer = Part_Graph(adj_matrix, self.upper_half_node, self.lower_half_node, in_dim, hidden_dim, cls_p,
                                     cls_h, cls_f)

    def forward(self, p_node_list, h_node_list, f_node_list, xp, xh, xf):
        # for full body node
        f_node_new_list = self.full_infer(f_node_list, h_node_list, p_node_list, xf)
        # for half body node
        h_node_list_new = self.half_infer(f_node_list, h_node_list, p_node_list, xh)
        # for part node
        p_node_list_new = self.part_infer(f_node_list, h_node_list, p_node_list, xp)

        return p_node_list_new, h_node_list_new, f_node_new_list

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

        # gnn infer
        self.gnn = GNN(adj_matrix, upper_half_node, lower_half_node, self.in_dim, self.hidden_dim, self.cls_p,
                       self.cls_h, self.cls_f)

        # node supervision
        self.node_seg = nn.Conv2d(hidden_dim, 1, 1)

    def forward(self, xp, xh, xf):
        # gnn inference at stride 8
        # feature transform
        f_node_list = list(torch.split(self.f_conv(xf), self.hidden_dim, dim=1))
        h_node_list = list(torch.split(self.h_conv(xh), self.hidden_dim, dim=1))
        p_node_list = list(torch.split(self.p_conv(xp), self.hidden_dim, dim=1))

        # node supervision
        f_seg = []
        h_seg = []
        p_seg = []
        f_seg.append(torch.cat([self.node_seg(node) for node in f_node_list], dim=1))
        h_seg.append(torch.cat([self.node_seg(node) for node in h_node_list], dim=1))
        p_seg.append(torch.cat([self.node_seg(node) for node in p_node_list], dim=1))

        # gnn infer
        p_node_list_new, h_node_list_new, f_node_list_new = self.gnn(p_node_list, h_node_list, f_node_list, xp, xh, xf)
        # node supervision new
        f_seg.append(torch.cat([self.node_seg(node) for node in f_node_list_new], dim=1))
        h_seg.append(torch.cat([self.node_seg(node) for node in h_node_list_new], dim=1))
        p_seg.append(torch.cat([self.node_seg(node) for node in p_node_list_new], dim=1))

        return p_seg, h_seg, f_seg

class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        self.layer5 = ASPPModule(2048, 512)
        self.layer_part = DecoderModule(num_classes)
        self.layer_half = DecoderModule(hbody_cls)
        self.layer_full = DecoderModule(fbody_cls)
        
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(256), nn.ReLU(inplace=False),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

        self.skip = nn.Sequential(nn.Conv2d(512, 512, kernel_size=1, padding=0, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False),
                                   )
        self.fuse = nn.Sequential(nn.Conv2d(1024, 512, kernel_size=3, padding=1, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))
        
        # adjacent matrix for pascal person 
        self.adj_matrix = torch.tensor(
            [[0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0]], requires_grad=False)
        
        # infer with hierarchical person graph
        self.gnn_infer = GNN_infer(adj_matrix=self.adj_matrix, upper_half_node=[1, 2, 3, 4], lower_half_node=[5, 6],
                                   in_dim=256, hidden_dim=32, cls_p=7, cls_h=3, cls_f=2)
        # aux layer
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(256), nn.ReLU(inplace=False),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        _,_,h,w = x[1].size()
        context = self.layer5(x[-1])
        context = F.interpolate(context, size=(h, w), mode='bilinear', align_corners=True)
        context = self.fuse(torch.cat([self.skip(x[1]), context], dim=1))

        p_fea = self.layer_part(context)
        h_fea = self.layer_half(context)
        f_fea = self.layer_full(context)

        # gnn infer
        p_seg, h_seg, f_seg = self.gnn_infer(p_fea, h_fea, f_fea)

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
    model = OCNet(Bottleneck, [3, 4, 23, 3], num_classes)  # 101
    return model
