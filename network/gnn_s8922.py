import functools

import torch
import torch.nn as nn
from torch.nn import functional as F

from inplace_abn.bn import InPlaceABNSync
from modules.com_mod import Bottleneck, ResGridNet, SEModule
from modules.parse_mod import ASPPModule2 as ASPPModule

BatchNorm2d = functools.partial(InPlaceABNSync, activation='none')
class ConvGRU(nn.Module):
    def __init__(self, input_dim, hidden_dim, kernel_size):
        super(ConvGRU, self).__init__()
        self.padding = kernel_size[0] // 2, kernel_size[1] // 2
        self.hidden_dim = hidden_dim
        self.conv_gates = nn.Conv2d(input_dim + hidden_dim, 2, kernel_size=1, padding=0, stride=1, bias=True)
        self.conv_can = nn.Sequential(
            nn.Conv2d(input_dim + hidden_dim, hidden_dim, kernel_size=kernel_size, padding=self.padding, stride=1, bias=False)
        )

        nn.init.orthogonal_(self.conv_gates.weight)
        nn.init.constant_(self.conv_gates.bias, 0.)

    def forward(self, input_tensor, h_cur):
        """
        :param self:
        :param input_tensor: (b, c, h, w)
            input is actually the target_model
        :param h_cur: (b, c_hidden, h, w)
            current hidden and cell states respectively
        :return: h_next,
            next hidden state
        """
        combined = torch.cat([input_tensor, h_cur], dim=1)
        combined_conv = self.conv_gates(combined)

        gamma, beta = torch.split(combined_conv, 1, dim=1)
        reset_gate = torch.sigmoid(gamma)
        update_gate = torch.sigmoid(beta)

        combined = torch.cat([input_tensor, reset_gate*h_cur], dim=1)
        cc_cnm = self.conv_can(combined)
        cnm = torch.tanh(cc_cnm)

        h_next = (1 - update_gate) * h_cur + update_gate * cnm
        return h_next
    
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
    
class Relation(nn.Module):
    def __init__(self, hidden_dim):
        super(Relation, self).__init__()
        self.relation = nn.Sequential(
            nn.Conv2d(2 * hidden_dim, hidden_dim, kernel_size=3, padding=1, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        )
        
    def forward(self, node, message):
        out = self.relation(torch.cat([node, message], dim=1))
        return out
    
       
class Composition(nn.Module):
    def __init__(self, hidden_dim, parts_num):
        super(Composition, self).__init__()
        self.relation = Relation(hidden_dim)
    def forward(self, parent, child_list):
        comp_message = self.relation(parent, sum(child_list))
        return comp_message
    
class Decomposition(nn.Module):
    def __init__(self, hidden_dim=10, child_num=2):
        super(Decomposition, self).__init__()
        self.relation = nn.ModuleList([Relation(hidden_dim) for i in range(child_num)])
        

    def forward(self, parent, child_list):
        decomp_list = [self.relation[i](child_list[i], parent-sum(child_list)+child_list[i]) for i in
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
    def __init__(self, in_dim=256, hidden_dim=10, parts_num = 6):
        super(Dep_Context, self).__init__()
        self.in_dim = in_dim
        self.hidden_dim = hidden_dim
        self.corrd_conv = nn.Sequential(
            nn.Conv2d(8, hidden_dim, kernel_size=1, padding=0, stride=1, bias=True)
        )
        self.query_conv = nn.Sequential(nn.Conv2d(hidden_dim+hidden_dim, hidden_dim, 1, bias=True))
        self.key_conv = nn.Sequential(nn.Conv2d(in_dim+hidden_dim, hidden_dim, 1, bias=True))
        # self.project = nn.Sequential(
        #     nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
        #     BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        # )
        self.query_conv = nn.ModuleList([self.query_conv for i in range(parts_num)])
        
        
        self.project = nn.ModuleList([nn.Sequential(
            nn.Conv2d(in_dim, hidden_dim, kernel_size=1, padding=0, stride=1, bias=False),
            BatchNorm2d(hidden_dim), nn.ReLU(inplace=False)
        ) for i in range(parts_num)])
        self.pool = nn.MaxPool2d(2)
    def forward(self, p_fea, hu_list):
        n, c, h, w = p_fea.size()
        p_fea0 = p_fea
        p_fea = self.pool(p_fea)
        n, c, hp, wp = p_fea.size()
        
        coord_fea = torch.from_numpy(generate_spatial_batch(hp,wp))
        coord_fea = coord_fea.to(p_fea.device).repeat((n, 1, 1, 1)).permute(0,3,1,2)
        coord_fea = self.corrd_conv(coord_fea)
        p_fea_coord = torch.cat([p_fea, coord_fea], dim=1)
        key = self.key_conv(p_fea_coord).view(n, -1, hp*wp) # n, hpwp, c+8
        dep_cont = []
        for i in range(len(hu_list)):
            # query = self.query_conv(p_fea_coord*self.pool(hu_att_list[i])).view(n, -1, hp*wp) # n, c, hw 
            query = self.query_conv[i](torch.cat([self.pool(hu_list[i]), coord_fea], dim=1)).view(n, -1, hp*wp).permute(0,2,1) # n, hpwp, c+8,
            energy = torch.bmm(query, key)  # n,hpwp,hpwp
            co_context = torch.bmm(energy,p_fea.view(n, -1, hp*wp).permute(0,2,1)).permute(0,2,1).view(n, -1, hp, wp)
            co_context = F.interpolate(co_context, (h,w), mode = 'bilinear', align_corners=True)
            co_context = self.project[i](co_context)
            dep_cont.append(co_context)
        return dep_cont


class Contexture(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, part_list_list=None):
        super(Contexture, self).__init__()
        self.hidden_dim =hidden_dim
        self.F_cont = Dep_Context(in_dim, hidden_dim, len(part_list_list))

        self.att_list = nn.ModuleList([nn.Sequential(
            nn.Conv2d(hidden_dim, len(part_list_list[i]), kernel_size=1, padding=0, stride=1, bias=True)
        ) for i in range(len(part_list_list))])
        self.F_dep_att_list = nn.ModuleList([nn.Sequential(
            nn.Conv2d(hidden_dim, 1, kernel_size=1, padding=0, stride=1, bias=True), nn.Sigmoid()
        ) for i in range(len(part_list_list))])
        self.part_list_list = part_list_list
        self.softmax = nn.Softmax(dim=1)

    def forward(self, p_list, p_fea):
        F_dep_list = self.F_cont(p_fea, p_list)
        F_dep_att_list = [self.F_dep_att_list[i](F_dep_list[i]) for i in range(len(p_list))]
        att_list = [self.att_list[i](F_dep_list[i]) for i in range(len(p_list))]
        
        att_list_list = [list(torch.split(F_dep_att_list[i]*self.softmax(att_list[i]), 1, dim=1)) for i in range(len(p_list))]
        return F_dep_list, att_list_list, att_list


class Dependency(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10):
        super(Dependency, self).__init__()
        self.relation = Relation(hidden_dim)
    def forward(self, hv, huv_context):
        dep_message = self.relation(hv, huv_context)
        return dep_message
    
class Full_Graph(nn.Module):
    def __init__(self, in_dim=256, hidden_dim=10, cls_p=7, cls_h=3, cls_f=2):
        super(Full_Graph, self).__init__()
        self.cls_f = cls_f
        self.comp_full = Composition(hidden_dim, cls_h-1)
        
        self.update = nn.ModuleList([ConvGRU(hidden_dim,hidden_dim,(1,1)) for i in range(cls_f-1)])

    def forward(self, f_node_list, h_node_list, p_node_list, xf):
        f_node_list_new = []
        for i in range(self.cls_f):
            if i==0:
                node = h_node_list[0]
            elif i==1:
                comp_full = self.comp_full(f_node_list[1], h_node_list[1:])
                node = self.update[i-1](comp_full, f_node_list[1])
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
        self.decomp_f = Decomposition(hidden_dim, 2)
        
        self.comp_u = Composition(hidden_dim, self.upper_parts_len)
        self.comp_l = Composition(hidden_dim, self.lower_parts_len)

        self.update = nn.ModuleList([ConvGRU(hidden_dim,hidden_dim,(1,1)) for i in range(cls_h-1)])

    def forward(self, f_node_list, h_node_list, p_node_list, xh):
        upper_parts = []
        for part in self.upper_part_list:
            upper_parts.append(p_node_list[part])

        lower_parts = []
        for part in self.lower_part_list:
            lower_parts.append(p_node_list[part])

        h_node_list_new = []
        decomp_f_list = self.decomp_f(f_node_list[1], h_node_list[1:])
        for i in range(self.cls_h):
            if i==0:
                node = h_node_list[0]
            elif i==1:
                comp = self.comp_u(h_node_list[i], upper_parts) 
                decomp = decomp_f_list[i-1]
                node = self.update[i-1](comp + decomp, h_node_list[i])
            elif i==2:
                comp = self.comp_l(h_node_list[i], lower_parts)
                decomp = decomp_f_list[i-1]
                node = self.update[i-1](comp + decomp, h_node_list[i])
            h_node_list_new.append(node)

        return h_node_list_new


class Part_Graph(nn.Module):
    def __init__(self, adj_matrix, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], in_dim=256, hidden_dim=10,
                 cls_p=7, cls_h=3, cls_f=2):
        super(Part_Graph, self).__init__()
        self.cls_p = cls_p
        self.hidden = hidden_dim
        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.edge_index = torch.nonzero(adj_matrix)
        self.edge_index_num = self.edge_index.shape[0]
        self.part_list_list = [[] for i in range(self.cls_p - 1)]
        for i in range(self.edge_index_num):
            self.part_list_list[self.edge_index[i, 1]].append(self.edge_index[i, 0])

        self.decomp_u = Decomposition(hidden_dim, len(upper_part_list))
        self.decomp_l = Decomposition(hidden_dim, len(lower_part_list))
        self.update = nn.ModuleList([ConvGRU(hidden_dim,hidden_dim,(1,1)) for i in range(cls_p-1)])

    def forward(self, f_node_list, h_node_list, p_node_list, xp):
        # upper half
        upper_parts = []
        for part in self.upper_part_list:
            upper_parts.append(p_node_list[part])
        # lower half
        lower_parts = []
        for part in self.lower_part_list:
            lower_parts.append(p_node_list[part])

        # decomposition
        p_node_list_new = []
        decomp_u_list = self.decomp_u(h_node_list[1], upper_parts)
        decomp_l_list = self.decomp_l(h_node_list[2], lower_parts)
        
        for i in range(self.cls_p):
            if i==0:
                node = p_node_list[0]
            elif i in self.upper_part_list:
                decomp = decomp_u_list[self.upper_part_list.index(i)]
                node = self.update[i-1](decomp, p_node_list[i])
            elif i  in self.lower_part_list:
                decomp = decomp_l_list[self.lower_part_list.index(i)]
                node = self.update[i-1](decomp, p_node_list[i])

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
        self.gnn = nn.ModuleList([GNN(adj_matrix, upper_half_node, lower_half_node, self.in_dim, self.hidden_dim, self.cls_p,
                       self.cls_h, self.cls_f) for i in range(2)])

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

        for iter in range(2):
            # gnn infer
            p_node_list_new, h_node_list_new, f_node_list_new = self.gnn[1](p_node_list, h_node_list, f_node_list, xp, xh, xf)
            # node supervision new
            f_seg.append(torch.cat([self.node_seg(node) for node in f_node_list_new], dim=1))
            h_seg.append(torch.cat([self.node_seg(node) for node in h_node_list_new], dim=1))
            p_seg.append(torch.cat([self.node_seg(node) for node in p_node_list_new], dim=1))
            p_node_list = p_node_list_new
            h_node_list = h_node_list_new
            f_node_list = f_node_list_new

        return p_seg, h_seg, f_seg


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
