from itertools import filterfalse as ifilterfalse

import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.autograd import Variable

from torch.nn import BCELoss
import utils.aaf.losses as lossx
class AAF_Loss(nn.Module):
    """
    Loss function for multiple outputs
    """

    def __init__(self, ignore_index=255,  only_present=True, num_classes=7):
        super(AAF_Loss, self).__init__()
        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.80777327, 1.00125961, 0.90997236, 1.10867908, 1.17541499,
                                         0.86041422, 1.01116758, 0.89290045, 1.12410812, 0.91105395,
                                         1.07604013, 1.12470610, 1.09895196, 0.90172057, 0.93529453,
                                         0.93054733, 1.04919178, 1.04937547, 1.06267568, 1.06365688])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

        self.num_classes=num_classes
        self.kld_margin=3.0
        self.kld_lambda_1=1.0
        self.kld_lambda_2=1.0
        self.dec = 1e-3
        # self.dec = 1e-2
        self.softmax = nn.Softmax(dim=1)
        self.w_edge = nn.Parameter(torch.zeros(1,1,1,self.num_classes,1,3))
        self.w_edge_softmax = nn.Softmax(dim=-1)
        self.w_not_edge = nn.Parameter(torch.zeros(1, 1, 1, self.num_classes, 1, 3))
        self.w_not_edge_softmax = nn.Softmax(dim=-1)

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        pred = F.interpolate(input=preds[0], size=(h, w), mode='bilinear', align_corners=True)
        pred = F.softmax(input=pred, dim=1)
        loss = lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present)
        # half body
        pred_hb = F.interpolate(input=preds[1], size=(h, w), mode='bilinear', align_corners=True)
        pred_hb = F.softmax(input=pred_hb, dim=1)
        loss_hb = lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present)
        # full body
        pred_fb = F.interpolate(input=preds[2], size=(h, w), mode='bilinear', align_corners=True)
        pred_fb = F.softmax(input=pred_fb, dim=1)
        loss_fb = lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present)
        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])



        #aaf loss
        labels = targets[0]
        one_label=labels.clone()
        one_label[one_label==255]=0
        one_hot_lab=F.one_hot(one_label, num_classes=self.num_classes)

        targets_p_node_list = list(torch.split(one_hot_lab,1, dim=3))
        for i in range(self.num_classes):
            targets_p_node_list[i] = targets_p_node_list[i].squeeze(-1)
            targets_p_node_list[i][labels==255]=255
        one_hot_lab = torch.stack(targets_p_node_list, dim=-1)

        prob = pred
        w_edge = self.w_edge_softmax(self.w_edge)
        w_not_edge = self.w_not_edge_softmax(self.w_not_edge)

        # w_edge_shape=list(w_edge.shape)
        # Apply AAF on 3x3 patch.
        eloss_1, neloss_1 = lossx.adaptive_affinity_loss(labels,
                                                         one_hot_lab,
                                                         prob,
                                                         1,
                                                         self.num_classes,
                                                         self.kld_margin,
                                                         w_edge[..., 0],
                                                         w_not_edge[..., 0])
        # Apply AAF on 5x5 patch.
        eloss_2, neloss_2 = lossx.adaptive_affinity_loss(labels,
                                                         one_hot_lab,
                                                         prob,
                                                         2,
                                                         self.num_classes,
                                                         self.kld_margin,
                                                         w_edge[..., 1],
                                                         w_not_edge[..., 1])
        # Apply AAF on 7x7 patch.
        eloss_3, neloss_3 = lossx.adaptive_affinity_loss(labels,
                                                         one_hot_lab,
                                                         prob,
                                                         3,
                                                         self.num_classes,
                                                         self.kld_margin,
                                                         w_edge[..., 2],
                                                         w_not_edge[..., 2])
        dec = self.dec
        aaf_loss = torch.mean(eloss_1) * self.kld_lambda_1 * dec
        aaf_loss += torch.mean(eloss_2) * self.kld_lambda_1 * dec
        aaf_loss += torch.mean(eloss_3) * self.kld_lambda_1 * dec
        aaf_loss += torch.mean(neloss_1) * self.kld_lambda_2 * dec
        aaf_loss += torch.mean(neloss_2) * self.kld_lambda_2 * dec
        aaf_loss += torch.mean(neloss_3) * self.kld_lambda_2 * dec

        # return torch.stack([loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn, aaf_loss], dim=0)
        return loss + 0.4 * loss_hb + 0.4 * loss_fb + 0.4 * loss_dsn + aaf_loss