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
            nn.Conv2d(2 * hidden_dim, 2*hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2*hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
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
            nn.Conv2d(2 * hidden_dim, 2*hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2*hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )

    def forward(self, parent, child_list, decomp_map):
        decomp_att = torch.softmax(decomp_map, dim=1)
        decomp_att_list = torch.split(decomp_att, 1, dim=1)
        decomp_list = [self.relation(torch.cat([parent * decomp_att_list[i+1], child_list[i]], dim=1)) for i in
                          range(len(child_list))]
        return decomp_list

class Contexture(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, part_list_list=None):
        super(Contexture, self).__init__()
        self.hidden_dim =hidden_dim
        self.att_list = nn.ModuleList([nn.Sequential(nn.Conv2d(hidden_dim*len(part_list_list[i]), 1, kernel_size=1, padding=0, stride=1, bias=True), nn.Sigmoid()) for i in range(len(part_list_list))])

        self.part_list_list = part_list_list

        self.project = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
            ) for i in range(len(part_list_list))])
    def forward(self, xp_list, p_fea):
        dep_context_list = []
        Fdep_att_list = []
        for i in range(len(xp_list)):
            context_node_list = torch.cat([xp_list[j] for j in self.part_list_list[i]], dim=1)
            att = self.att_list[i](context_node_list)
            Fdep_att_list.append(att)
            dep_context_list.append(self.project[i](p_fea*(2-att)))

        return Fdep_att_list, dep_context_list


class Dependency(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10):
        super(Dependency, self).__init__()
        self.relation = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, 2*hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(2*hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )
    def forward(self, hv, dep_hv):
        dep_message = self.relation(torch.cat([dep_hv, hv], dim=1))
        return dep_message


class conv_Update(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10):
        super(conv_Update, self).__init__()
        self.hidden_dim = hidden_dim
        dtype = torch.cuda.FloatTensor
        self.update = ConvGRU(input_dim=hidden_dim,
                              hidden_dim=hidden_dim,
                              kernel_size=(3, 3),
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
        self.conv0 = nn.Sequential(nn.Conv2d(512, 512, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(512), nn.ReLU(inplace=False))
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, dilation=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False))
        self.alpha = nn.Parameter(torch.ones(1))

    def forward(self, xt, xm):
        _, _, h, w = xm.size()
        xt = self.conv0(F.interpolate(xt, size=(h, w), mode='bilinear', align_corners=True) + self.alpha * xm)
        xt_fea = self.conv1(xt)
        return xt_fea

class AlphaDecoder(nn.Module):
    def __init__(self, hbody_cls):
        super(AlphaDecoder, self).__init__()
        self.conv1 = nn.Sequential(nn.Conv2d(512, 256, kernel_size=3, padding=1, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False),
                                   nn.Conv2d(256, 256, kernel_size=1, padding=0, stride=1, bias=False),
                                   BatchNorm2d(256), nn.ReLU(inplace=False), 
                                   SEModule(256, reduction=16)
                                   )
        self.alpha_hb = nn.Parameter(torch.ones(1))

    def forward(self, x, skip):
        _, _, h, w = skip.size()

        xup = F.interpolate(x, size=(h, w), mode='bilinear', align_corners=True)
        xfuse = xup + self.alpha_hb * skip
        output = self.conv1(xfuse)
        return output


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

        # part_dp_list = []
        Fdep_att_list, part_dp_list = self.F_dep_list(p_node_list, xp)
        # for i in range(self.cls_p-1):
            # part_dp_list.append(self.part_dp(p_node_list[i], dep_node_list[i]))
        
        xp_list_new = []
        for i in range(self.cls_p - 1):
            if i + 1 in self.upper_part_list:
                message = decomp_pu_list[self.upper_part_list.index(i + 1)] + part_dp_list[i]
            elif i + 1 in self.lower_part_list:
                message = decomp_pl_list[self.lower_part_list.index(i + 1)] + part_dp_list[i]

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
        self.final_cls = Final_cls(48, hidden_dim, self.cls_p)

    def forward(self, xp, xh, xf, xl):
        # gnn inference at stride 8
        # feature transform
        f_node_list = list(torch.split(self.f_conv(xf), self.hidden_dim, dim=1))
        p_node_list = list(torch.split(self.p_conv(xp), self.hidden_dim, dim=1))
        h_node_list = list(torch.split(self.h_conv(xh), self.hidden_dim, dim=1))

        # node supervision
        f_seg = self.f_seg(torch.cat(f_node_list, dim=1))
        h_seg = self.h_seg(torch.cat(h_node_list, dim=1))
        p_seg = self.p_seg(torch.cat(p_node_list, dim=1))

        # gnn infer
        p_node_list_new, h_node_list_new, f_node_new, decomp_map_f, decomp_map_u, decomp_map_l, comp_map_f, comp_map_u, comp_map_l, Fdep_att_list = self.gnn(p_node_list[1:], h_node_list[1:], f_node_list[1], xp, xh, xf)

        # node supervision new
        f_seg_new = self.f_seg(torch.cat([f_node_list[0], f_node_new], dim=1))
        h_seg_new = self.h_seg(torch.cat([h_node_list[0]]+h_node_list_new, dim=1))
        p_seg_new = self.p_seg(torch.cat([p_node_list[0]]+p_node_list_new, dim=1))

        #final readout
        # p_seg_final = self.final_cls([p_node_list[0]]+p_node_list_new, xl)
        return [p_seg, p_seg_new], [h_seg, h_seg_new], [f_seg, f_seg_new], [decomp_map_f], [decomp_map_u], [
            decomp_map_l], [comp_map_f], [comp_map_u], [comp_map_l], [Fdep_att_list]

# class Final_cls(nn.Module):
#     def __init__(self, in_dim=256, num_classes=7):
#         super(Final_cls, self).__init__()

#         self.query_conv = nn.Conv1d(in_dim, in_dim//4, kernel_size=1)
#         self.key_conv = nn.Conv2d(in_dim, in_dim//4, kernel_size=1)
#         self.cls_conv = nn.Sequential(
#             nn.Dropout2d(0.1),
#             nn.Conv2d(in_dim+in_dim, num_classes, kernel_size=1, padding=0, bias=True)
#         )

#     def forward(self, p_seg, p_fea):
#         # n, c, h, w = p_seg.size()
#         n, _, h, w = p_fea.size()
#         p_seg = F.interpolate(p_seg, (h, w), mode='bilinear', align_corners=True)
#         p_att = torch.softmax(p_seg, dim=1).view(n, -1, h*w).permute(0,2,1) # n, h*w, c
#         p_center = torch.bmm(p_fea.view(n, -1, h*w),p_att)/torch.sum(p_att, dim=1, keepdim=True) #n, C, c

#         query = self.query_conv(p_center) # n, C', c
#         key = self.key_conv(p_fea).view(n, -1, h*w) # n, C', h*w
        
#         energy = torch.bmm(query.permute(0,2,1), key) #n, c, h*w
#         attention = torch.softmax(energy, dim=1)
#         new_fea = torch.bmm(p_center, attention).view(n, -1, h, w) #n, C, h*w
#         new_seg = self.cls_conv(torch.cat([p_fea, new_fea], dim=1))
#         return new_seg

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
class Final_cls(nn.Module):
    def __init__(self, in_dim, hidden_dim, num_classes):
        super(Final_cls, self).__init__()
        self.num_classes = num_classes
        self.cls_conv = nn.Sequential(
            nn.Conv2d(num_classes*(in_dim+hidden_dim), num_classes*hidden_dim, kernel_size=3, padding=1, groups=num_classes, bias=False), 
            BatchNorm2d(num_classes*hidden_dim), nn.ReLU(inplace=False),
            nn.Conv2d(num_classes*hidden_dim, num_classes*hidden_dim, kernel_size=1, padding=0, groups=num_classes, bias=False), 
            BatchNorm2d(num_classes*hidden_dim), nn.ReLU(inplace=False),
            nn.Dropout2d(0.1),
            nn.Conv2d(num_classes*hidden_dim, num_classes, kernel_size=1, padding=0, groups=num_classes, bias=True)
        )
        self.conv2 = nn.Sequential(nn.Conv2d(256, in_dim, kernel_size=1, stride=1, padding=0, dilation=1, bias=False),
                                   BatchNorm2d(in_dim), nn.ReLU(inplace=False))

    def forward(self, p_node_list, xl):
        _, _, h, w = xl.size()
        xl = self.conv2(xl)
        p_node_list_final = [torch.cat([F.interpolate(p_node_list[i], (h, w), mode='bilinear', align_corners=True), xl], dim=1) for i in range(self.num_classes)]
        new_score = self.cls_conv(torch.cat(p_node_list_final, dim=1))
        return new_score

class Decoder(nn.Module):
    def __init__(self, num_classes=7, hbody_cls=3, fbody_cls=2):
        super(Decoder, self).__init__()
        # self.layer5 = MagicModule(2048, 512, 1)
        self.layer5 = ASPPModule(2048, 512)
        self.layer6 = DecoderModule(num_classes)
        self.layerh = AlphaDecoder(hbody_cls)
        self.layerf = AlphaDecoder(fbody_cls)
        
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
        context = self.layer5(x[-1])

        # direct infer
        p_fea = self.layer6(context, x[1])
        h_fea = self.layerh(context, x[1])
        f_fea = self.layerf(context, x[1])

        # gnn infer
        p_seg, h_seg, f_seg, decomp_map_f, decomp_map_u, decomp_map_l, comp_map_f, comp_map_u, comp_map_l, \
        Fdep_att_list= self.gnn_infer(p_fea, h_fea, f_fea, x[0])

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
