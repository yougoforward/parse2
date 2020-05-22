import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import MagicModule, ASPPModule
from modules.senet import se_resnext50_32x4d, se_resnet101, senet154

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
from modules.convGRU import ConvGRU
from modules.dcn import DFConv2d

class Comp_att(nn.Module):
    def __init__(self, hidden_dim, parts_num):
        super(Comp_att, self).__init__()
        self.comp_att = nn.Sequential(
            nn.Conv2d(parts_num * hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True),
            nn.Sigmoid()
        )
    def forward(self, child_list):
        comp_att = self.comp_att(torch.cat(child_list, dim=1))
        return comp_att

class Composition(nn.Module):
    def __init__(self, hidden_dim):
        super(Composition, self).__init__()
        self.relation = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )
    def forward(self, parent, child_list, comp_att):
        comp_message = sum([self.relation(torch.cat([parent, child * comp_att], dim=1)) for child in child_list])
        return comp_message

class Decomp_att(nn.Module):
    def __init__(self, hidden_dim, parts_num):
        super(Decomp_att, self).__init__()
        self.decomp_map = nn.Sequential(
            nn.Conv2d(hidden_dim, parts_num+1, kernel_size=1, padding=0, stride=1, bias=True)
        )
    def forward(self, parent):
        decomp_map = self.decomp_map(parent)
        return decomp_map


class Decomposition(nn.Module):
    def __init__(self, hidden_dim=10):
        super(Decomposition, self).__init__()
        self.relation = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )

    def forward(self, parent, child_list, decomp_map):
        decomp_att = torch.softmax(decomp_map, dim=1)
        decomp_att_list = torch.split(decomp_att, 1, dim=1)
        decomp_list = [self.relation(torch.cat([parent * decomp_att_list[i+1], child_list[i]], dim=1)) for i in
                          range(len(child_list))]
        return decomp_list


def generate_spatial_batch(featmap_H, featmap_W):
    import numpy as np
    spatial_batch_val = np.zeros((1, featmap_H, featmap_W, 8), dtype=np.float32)
    for h in range(featmap_H):
        for w in range(featmap_W):
            xmin = w / featmap_W * 2 - 1
            xmax = (w + 1) / featmap_W * 2 - 1
            xctr = (xmin + xmax) / 2
            ymin = h / featmap_H * 2 - 1
            ymax = (h + 1) / featmap_H * 2 - 1
            yctr = (ymin + ymax) / 2
            spatial_batch_val[:, h, w, :] = \
                [xmin, ymin, xmax, ymax, xctr, yctr, 1 / featmap_W, 1 / featmap_H]
    return spatial_batch_val



class Dep_Context(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10):
        super(Dep_Context, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.softmax = nn.Softmax(dim=-1)

        self.query_conv = nn.Sequential(nn.Conv2d(hidden_dim+8, hidden_dim+8, 1, bias=False))

        self.key_conv = nn.Sequential(nn.Conv2d(in_dim+8, hidden_dim+8, 1, bias=False))

        self.value_conv = nn.Sequential(nn.Conv2d(in_dim, 128, 1, bias=False),
                                     BatchNorm2d(128), nn.ReLU(inplace=False))


        self.project = nn.Sequential(nn.Conv2d(128, hidden_dim, 1, bias=False),
                                     BatchNorm2d(hidden_dim), nn.ReLU(inplace=False))
        self.pool = nn.AvgPool2d(3, 2)

    def forward(self, p_fea, hu):
        n, c, h, w = p_fea.size()
        p_fea = self.pool(p_fea)
        hu = self.pool(hu)
        _, _, hp, wp = p_fea.size()

        coord_fea = torch.from_numpy(generate_spatial_batch(hp,wp))
        coord_fea = coord_fea.to(p_fea.device).repeat((n, 1, 1, 1)).permute(0,3,1,2)

        query = self.query_conv(torch.cat([hu, coord_fea], dim=1)).view(n, self.hidden_dim+8, -1).permute(0, 2, 1) # n, h*w, hid+8
        key = self.key_conv(torch.cat([p_fea, coord_fea], dim=1)).view(n, self.hidden_dim+8, -1)
        val = self.value_conv(p_fea)
        
        energy = torch.bmm(query, key)  # n,hw,hw
        attention = self.softmax(energy)

        co_context = torch.bmm(val.view(n, 128, -1), attention.permute(0,2,1)).view(n, 128, hp, wp)
        co_context = self.project(co_context)
        co_context = F.interpolate(co_context, (h, w), mode='bilinear', align_corners=True)
        return co_context


class Contexture(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, part_list_list=None):
        super(Contexture, self).__init__()
        self.hidden_dim =hidden_dim
        self.F_cont = nn.ModuleList(
            [Dep_Context(in_dim, hidden_dim) for i in range(len(part_list_list))])

        self.att_list = nn.ModuleList([nn.Conv2d(hidden_dim, len(part_list_list[i])+ 1, kernel_size=1, padding=0, stride=1, bias=True)
                                       for i in range(len(part_list_list))])

        self.context_att_list = nn.ModuleList([nn.Sequential(
            nn.Conv2d(2*hidden_dim, 2, kernel_size=1, padding=0, stride=1, groups=2, bias=True)
        ) for i in range(len(part_list_list))])

        self.softmax = nn.Softmax(dim=1)

    def forward(self, xp_list, p_fea):
        context_list = []
        F_dep_list =[]
        for i in range(len(xp_list)):
            context = self.F_cont[i](p_fea, xp_list[i])
            F_dep_list.append(context)

        att_list = [self.att_list[i](F_dep_list[i]) for i in range(len(xp_list))]
        att_list_list = [list(torch.split(self.softmax(att_list[i]), 1, dim=1)) for i in range(len(xp_list))]
        return F_dep_list, att_list_list, att_list


class Dependency(nn.Module):
    def __init__(self, hidden_dim=10):
        super(Dependency, self).__init__()
        self.relation = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )
    def forward(self, hv, hu_context, dep_att_huv):
        dep_message = self.relation(torch.cat([hu_context*dep_att_huv, hv], dim=1))
        return dep_message


class conv_Update(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10):
        super(conv_Update, self).__init__()
        self.hidden_dim = hidden_dim
        dtype = torch.cuda.FloatTensor
        self.update = ConvGRU(input_dim=hidden_dim,
                              hidden_dim=hidden_dim,
                              kernel_size=(1, 1),
                              num_layers=1,
                              dtype=dtype,
                              batch_first=True,
                              bias=True,
                              return_all_layers=False)

    def forward(self, x, h, message):
        _, out = self.update(message.unsqueeze(1), [h])
        return out[0][0]


class DecoderModule(nn.Module):

    def __init__(self, num_classes):
        super(DecoderModule, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(256, 48, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(48), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(512+48, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16))


    def forward(self, xt, xm, xl):
        _, _, th, tw = xl.size()
        xt_up = F.interpolate(xt, size=(th, tw), mode='bilinear', align_corners=True)
        x_skip = self.conv0(xl)
        xt_fea = self.conv1(torch.cat([xt_up, x_skip], dim=1))
        return xt_fea


class BetaHBDecoder(nn.Module):
    def __init__(self, hbody_cls):
        super(BetaHBDecoder, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(512, 96, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(96), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(512+96, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16))

    def forward(self, x, skip):
        _, _, h, w = skip.size()
        x_skip = self.conv0(skip)
        x_up = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = self.conv1(torch.cat([x_up, x_skip], dim=1))
        return xfuse


class AlphaFBDecoder(nn.Module):
    def __init__(self, fbody_cls):
        super(AlphaFBDecoder, self).__init__()
        self.conv0 = nn.Sequential(nn.Conv2d(512, 96, kernel_size=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(96), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(512+96, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), SEModule(256, reduction=16))

    def forward(self, x, skip):
        _, _, h, w = skip.size()
        x_skip = self.conv0(skip)
        x_up = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = self.conv1(torch.cat([x_up, x_skip], dim=1))
        return xfuse


class Full_Graph(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Full_Graph, self).__init__()
        self.hidden = hidden_dim
        self.comp_h = Composition(hidden_dim)
        self.comp_att = Comp_att(hidden_dim, cls_h-1)
        self.conv_Update = conv_Update(in_dim, hidden_dim)

    def forward(self, f_node, h_node_list, p_node_list, xf):
        comp_map_f = self.comp_att(h_node_list)
        comp_h = self.comp_h(f_node, h_node_list, comp_map_f)
        f_node_new = self.conv_Update(xf, f_node, comp_h)
        return f_node_new, comp_map_f


class Half_Graph(nn.Module):
    def __init__(self, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], in_dim=256, hidden_dim=10, cls_p=7,
                 cls_h=3, cls_f=2):
        super(Half_Graph, self).__init__()
        self.cls_h = cls_h
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.upper_parts_len = len(upper_part_list)
        self.lower_parts_len = len(lower_part_list)
        self.hidden = hidden_dim
        self.decomp_att = Decomp_att(hidden_dim, cls_h-1)
        self.decomp_fh_list = Decomposition(hidden_dim)
        self.comp_att_u = Comp_att(hidden_dim, self.upper_parts_len)
        self.comp_att_l = Comp_att(hidden_dim, self.lower_parts_len)
        self.comp = Composition(hidden_dim)

        self.update_u = conv_Update(in_dim, hidden_dim)
        self.update_l = conv_Update(in_dim, hidden_dim)

    def forward(self, f_node, h_node_list, p_node_list, xh):
        # decomposition full node to half node
        decomp_map = self.decomp_att(f_node)
        decomp_list = self.decomp_fh_list(f_node, h_node_list, decomp_map)

        # composition part node to half node
        # upper half
        upper_parts = []
        for part in self.upper_part_list:
            upper_parts.append(p_node_list[part - 1])
        comp_map_u = self.comp_att_u(upper_parts)

        comp_u = self.comp(h_node_list[0], upper_parts, comp_map_u)
        message_u = decomp_list[0] + comp_u
        xh_u = self.update_u(xh, h_node_list[0], message_u)

        # lower half
        lower_parts = []
        for part in self.lower_part_list:
            lower_parts.append(p_node_list[part - 1])
        comp_map_l = self.comp_att_l(lower_parts)

        comp_l = self.comp(h_node_list[1], lower_parts, comp_map_l)
        message_l = decomp_list[1] + comp_l
        xh_l = self.update_l(xh, h_node_list[1], message_l)

        xh_list_new = [xh_u, xh_l]
        return xh_list_new, decomp_map, comp_map_u, comp_map_l


class Part_Graph(nn.Module):
    def __init__(self, adj_matrix, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], in_dim=256, hidden_dim=10,
                 cls_p=7, cls_h=3, cls_f=2):
        super(Part_Graph, self).__init__()
        self.cls_p = cls_p
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.upper_parts_len = len(upper_part_list)
        self.lower_parts_len = len(lower_part_list)

        self.edge_index = torch.nonzero(adj_matrix)
        self.edge_index_num = self.edge_index.shape[0]
        self.part_list_list = [[] for i in range(self.cls_p - 1)]
        for i in range(self.edge_index_num):
            self.part_list_list[self.edge_index[i, 1]].append(self.edge_index[i, 0])

        self.decomp_att_u = Decomp_att(hidden_dim, self.upper_parts_len)
        self.decomp_att_l = Decomp_att(hidden_dim, self.lower_parts_len)
        self.decomp_hp = Decomposition(hidden_dim)

        self.F_dep_list = Contexture(in_dim=in_dim, hidden_dim=hidden_dim, part_list_list=self.part_list_list)

        self.part_dp = Dependency(hidden_dim)

        self.node_update_list = nn.ModuleList([conv_Update(in_dim, hidden_dim) for i in range(self.cls_p - 1)])

    def forward(self, f_node, h_node_list, p_node_list, xp):
        # upper half
        upper_parts = []
        for part in self.upper_part_list:
            upper_parts.append(p_node_list[part - 1])
        # lower half
        lower_parts = []
        for part in self.lower_part_list:
            lower_parts.append(p_node_list[part - 1])

        decomp_map_u = self.decomp_att_u(h_node_list[0])
        decomp_map_l = self.decomp_att_l(h_node_list[1])
        decomp_pu_list = self.decomp_hp(h_node_list[0], upper_parts, decomp_map_u)
        decomp_pl_list = self.decomp_hp(h_node_list[1], lower_parts, decomp_map_l)

        F_dep_list, att_list_list, Fdep_att_list = self.F_dep_list(p_node_list, xp)

        xpp_list_list = [[] for i in range(self.cls_p - 1)]
        for i in range(self.edge_index_num):
            xpp_list_list[self.edge_index[i, 1]].append(
                self.part_dp(p_node_list[self.edge_index[i, 1]], 
                F_dep_list[self.edge_index[i, 0]], 
                att_list_list[self.edge_index[i, 0]][1+self.part_list_list[self.edge_index[i, 0]].index(self.edge_index[i, 1])]))
        
        xp_list_new = []
        for i in range(self.cls_p - 1):
            if i + 1 in self.upper_part_list:
                message = decomp_pu_list[self.upper_part_list.index(i + 1)] + sum(xpp_list_list[i])
            elif i + 1 in self.lower_part_list:
                message = decomp_pl_list[self.lower_part_list.index(i + 1)] + sum(xpp_list_list[i])

            xp_list_new.append(self.node_update_list[i](xp, p_node_list[i], message))
        return xp_list_new, decomp_map_u, decomp_map_l, Fdep_att_list


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

    def forward(self, p_node_list, h_node_list, f_node, xp, xh, xf):
        # for full body node
        f_node_new, comp_map_f = self.full_infer(f_node, h_node_list, p_node_list, xf)
        # for half body node
        h_node_list_new, decomp_map_f, comp_map_u, comp_map_l = self.half_infer(f_node, h_node_list, p_node_list, xh)
        # for part node
        p_node_list_new, decomp_map_u, decomp_map_l, Fdep_att_list = self.part_infer(f_node, h_node_list, p_node_list, xp)

        return p_node_list_new, h_node_list_new, f_node_new, decomp_map_f, decomp_map_u, decomp_map_l, comp_map_f, comp_map_u, comp_map_l, Fdep_att_list


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
        # multi-label classifier
        self.f_seg = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(hidden_dim * cls_f, cls_f, 1, groups=cls_f))
        self.h_seg = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(hidden_dim * cls_h, cls_h, 1, groups=cls_h))
        self.p_seg = nn.Sequential(nn.Dropout2d(0.1), nn.Conv2d(hidden_dim * cls_p, cls_p, 1, groups=cls_p))

        #final seg
        self.final_cls = Final_cls(in_dim, self.cls_p)
        self.down = nn.Sequential(nn.Conv2d(256, 256, kernel_size=3, stride=2, padding=1, dilation=1, bias=False), BatchNorm2d(256), nn.ReLU(inplace=False))

    def forward(self, xp, xh, xf):
        # _, _, th, tw = xp.size()
        # _, _, h, w = xh.size()
        #
        # xh = F.interpolate(xh, (th, tw), mode='bilinear', align_corners=True)
        # xf = F.interpolate(xf, (th, tw), mode='bilinear', align_corners=True)
        xp_down = self.down(xp)
        # gnn inference at stride 8
        # feature transform
        f_node_list = list(torch.split(self.f_conv(xf), self.hidden_dim, dim=1))
        p_node_list = list(torch.split(self.p_conv(xp_down), self.hidden_dim, dim=1))
        h_node_list = list(torch.split(self.h_conv(xh), self.hidden_dim, dim=1))

        # node supervision
        f_seg = self.f_seg(torch.cat(f_node_list, dim=1))
        h_seg = self.h_seg(torch.cat(h_node_list, dim=1))
        p_seg = self.p_seg(torch.cat(p_node_list, dim=1))

        # gnn infer
        p_node_list_new, h_node_list_new, f_node_new, decomp_map_f, decomp_map_u, decomp_map_l, comp_map_f, comp_map_u, comp_map_l, Fdep_att_list = self.gnn(p_node_list[1:], h_node_list[1:], f_node_list[1], xp_down, xh, xf)

        # node supervision new
        f_seg_new = self.f_seg(torch.cat([f_node_list[0], f_node_new], dim=1))
        h_seg_new = self.h_seg(torch.cat([h_node_list[0]]+h_node_list_new, dim=1))
        p_seg_new = self.p_seg(torch.cat([p_node_list[0]]+p_node_list_new, dim=1))

        #final readout
        p_seg_final = self.final_cls(p_seg_new, xp)
        return [p_seg, p_seg_new, p_seg_final], [h_seg, h_seg_new], [f_seg, f_seg_new], [decomp_map_f], [decomp_map_u], [
            decomp_map_l], [comp_map_f], [comp_map_u], [comp_map_l], [Fdep_att_list]

class Final_cls(nn.Module):
    def __init__(self, in_dim=256, num_classes=7):
        super(Final_cls, self).__init__()

        self.query_conv = nn.Conv1d(in_dim, in_dim//4, kernel_size=1)
        self.key_conv = nn.Conv2d(in_dim, in_dim//4, kernel_size=1)
        self.cls_conv = nn.Sequential(
            nn.Dropout2d(0.1),
            nn.Conv2d(in_dim+in_dim, num_classes, kernel_size=1, padding=0, bias=True)
        )

    def forward(self, p_fea, p_seg):
        # n, c, h, w = p_seg.size()
        _, _, h, w = p_fea.size()
        p_seg = F.interpolate(p_seg, (h, w), mode='bilinear', align_corners=True)
        
        p_att = torch.softmax(p_seg, dim=1).view(n, -1, h*w).permute(0,2,1) # n, h*w, c
        p_center = torch.bmm(p_fea.view(n, -1, h*w),p_att)/torch.sum(p_att, dim=1, keepdim=True) #n, C, c

        query = self.query_conv(p_center) # n, C', c
        key = self.key_conv(p_fea).view(n, -1, h*w) # n, C', h*w
        
        energy = torch.bmm(query.permute(0,2,1), key) #n, c, h*w
        attention = torch.softmax(energy, dim=1)
        new_fea = torch.bmm(p_center, attention).view(n, -1, h, w) #n, C, h*w
        new_seg = self.cls_conv(torch.cat([p_fea, new_fea], dim=1))
        return new_seg

# class Final_cls(nn.Module):
#     def __init__(self, in_dim, num_classes):
#         super(Final_cls, self).__init__()

#         self.cls_conv = nn.Sequential(
#             nn.Conv2d(in_dim+num_classes, in_dim, kernel_size=3, padding=1, bias=False), 
#             BatchNorm2d(in_dim), nn.ReLU(inplace=False),
#             nn.Conv2d(in_dim, in_dim, kernel_size=1, padding=0, bias=False), 
#             BatchNorm2d(in_dim), nn.ReLU(inplace=False),
#             nn.Dropout2d(0.1),
#             nn.Conv2d(in_dim, num_classes, kernel_size=1, padding=0, bias=True)
#         )

#     def forward(self, xp, score):
#         _, _, h, w = xp.size()
#         up_score = F.interpolate(score, (h, w), mode='bilinear', align_corners=True)
#         new_score = self.cls_conv(torch.cat([xp, up_score], dim=1))
#         return new_score

class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        # self.layer5 = MagicModule(2048, 512, 1)
        self.layer5 = ASPPModule(2048, 512)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = BetaHBDecoder(hbody_cls)
        self.layerf = AlphaFBDecoder(fbody_cls)
        
        # adjacent matrix for pascal person 
        self.adj_matrix = torch.tensor(
            [[0, 1, 0, 0, 0, 0], [1, 0, 1, 0, 1, 0], [0, 1, 0, 1, 0, 0], [0, 0, 1, 0, 0, 0], [0, 1, 0, 0, 0, 1],
             [0, 0, 0, 0, 1, 0]], requires_grad=False)
        
        # infer with hierarchical person graph
        self.gnn_infer = GNN_infer(adj_matrix=self.adj_matrix, upper_half_node=[1, 2, 3, 4], lower_half_node=[5, 6],
                                   in_dim=256, hidden_dim=64, cls_p=7, cls_h=3, cls_f=2)
        # aux layer
        self.layer_dsn = nn.Sequential(nn.Conv2d(1024, 256, kernel_size=3, stride=1, padding=1),
                                       BatchNorm2d(256), nn.ReLU(inplace=False),
                                       nn.Conv2d(256, num_classes, kernel_size=1, stride=1, padding=0, bias=True))

    def forward(self, x):
        x_dsn = self.layer_dsn(x[-2])
        seg = self.layer5(x[-1])

        # direct infer
        x_fea = self.layer6(seg, x[1], x[0])
        alpha_hb_fea = self.layerh(seg, x[1])
        alpha_fb_fea = self.layerf(seg, x[1])

        # gnn infer
        p_seg, h_seg, f_seg, decomp_map_f, decomp_map_u, decomp_map_l, comp_map_f, comp_map_u, comp_map_l, \
        Fdep_att_list= self.gnn_infer(x_fea, alpha_hb_fea, alpha_fb_fea)

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
    # model = OCNet(Bottleneck, [3, 4, 6, 3], num_classes) #50
    model = OCNet(Bottleneck, [3, 4, 23, 3], num_classes)  # 101
    # model = OCNet(Bottleneck, [3, 8, 36, 3], num_classes)  #152
    return model
