class gnn_loss_dp(nn.Module):
    """Lovasz loss for Alpha process"""

    def __init__(self, adj_matrix, ignore_index=None, only_present=True, upper_part_list=[1, 2, 3, 4], lower_part_list=[5, 6], cls_p=7, cls_h=3, cls_f=2):
        super(gnn_loss_dp, self).__init__()
        self.edge_index = torch.nonzero(adj_matrix)
        self.edge_index_num = self.edge_index.shape[0]
        self.part_list_list = [[] for i in range(cls_p - 1)]
        for i in range(self.edge_index_num):
            self.part_list_list[self.edge_index[i, 1]].append(self.edge_index[i, 0])

        self.ignore_index = ignore_index
        self.only_present = only_present
        self.weight = torch.FloatTensor([0.82877791, 0.95688253, 0.94921949, 1.00538108, 1.0201687,  1.01665831, 1.05470914])
        self.criterion = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)
        self.criterion2 = torch.nn.CrossEntropyLoss(ignore_index=ignore_index, weight=None)

        self.upper_part_list = upper_part_list
        self.lower_part_list = lower_part_list
        self.num_classes = cls_p
        self.cls_h = cls_h
        self.cls_f = cls_f
        self.bceloss = torch.nn.BCELoss(reduction='mean')

    def forward(self, preds, targets):
        h, w = targets[0].size(1), targets[0].size(2)
        # seg loss
        loss=[]
        for i in range(len(preds[0])):
            pred = F.interpolate(input=preds[0][i], size=(h, w), mode='bilinear', align_corners=True)
            loss_ce = 0.4*self.criterion(pred, targets[0])
            pred = F.softmax(input=pred, dim=1)
            loss.append(loss_ce + lovasz_softmax_flat(*flatten_probas(pred, targets[0], self.ignore_index), only_present=self.only_present))

        # loss = sum(loss)
        # half body
        loss_hb = []
        for i in range(len(preds[1])):
            pred_hb = F.interpolate(input=preds[1][i], size=(h, w), mode='bilinear', align_corners=True)
            pred_hb = F.softmax(input=pred_hb, dim=1)
            loss_hb.append(lovasz_softmax_flat(*flatten_probas(pred_hb, targets[1], self.ignore_index),
                                      only_present=self.only_present))
        # loss_hb = sum(loss_hb)


        # full body
        loss_fb = []
        for i in range(len(preds[2])):
            pred_fb = F.interpolate(input=preds[2][i], size=(h, w), mode='bilinear', align_corners=True)
            pred_fb = F.softmax(input=pred_fb, dim=1)
            loss_fb.append(lovasz_softmax_flat(*flatten_probas(pred_fb, targets[2], self.ignore_index),
                                      only_present=self.only_present))
        # loss_fb = sum(loss_fb)

        #one hot part
        labels_p = targets[0]
        one_label_p = labels_p.clone().long()
        one_label_p[one_label_p == self.ignore_index] = 0
        one_hot_lab_p = F.one_hot(one_label_p, num_classes=self.num_classes)
        one_hot_pb_list = list(torch.split(one_hot_lab_p, 1, dim=-1))
        for i in range(0, self.num_classes):
            one_hot_pb_list[i] = one_hot_pb_list[i].squeeze(-1)
            # one_hot_pb_list[i][targets[0]==255]=255

        #one hot half
        labels_h = targets[1]
        one_label_h = labels_h.clone().long()
        one_label_h[one_label_h == self.ignore_index] = 0
        one_hot_lab_h = F.one_hot(one_label_h, num_classes=self.cls_h)
        one_hot_hb_list = list(torch.split(one_hot_lab_h, 1, dim=-1))
        for i in range(0, self.cls_h):
            one_hot_hb_list[i] = one_hot_hb_list[i].squeeze(-1)
            # one_hot_hb_list[i][targets[1]==255]=255

        #one hot full
        labels_f = targets[2]
        one_label_f = labels_f.clone().long()
        one_label_f[one_label_f == self.ignore_index] = 0
        one_hot_lab_f = F.one_hot(one_label_f, num_classes=self.cls_f)
        one_hot_fb_list = list(torch.split(one_hot_lab_f, 1, dim=-1))
        for i in range(0, self.cls_f):
            one_hot_fb_list[i] = one_hot_fb_list[i].squeeze(-1)
            # one_hot_fb_list[i][targets[2]==255]=255


        # #
        valid = (targets[0] != self.ignore_index).unsqueeze(1)

        #decomp fh
        target_fh = one_hot_hb_list[2]+(1-one_hot_fb_list[1])*self.ignore_index

        loss_fh_att = []
        for i in range(len(preds[3])):
            pred_fh = F.interpolate(input=preds[3][i], size=(h, w), mode='bilinear', align_corners=True)
            loss_fh_att.append(self.criterion2(pred_fh, target_fh.long()))
        loss_fh_att = sum(loss_fh_att)

        #decomp up
        upper_bg_node = 1-one_hot_hb_list[1]
        upper_parts=[]
        for i in self.upper_part_list:
            upper_parts.append(one_hot_pb_list[i])
        targets_up = torch.stack(upper_parts, dim=1)
        targets_up = targets_up.argmax(dim=1, keepdim=False)
        targets_up[upper_bg_node == 1] = self.ignore_index
        loss_up_att = []
        for i in range(len(preds[4])):
            pred_up = F.interpolate(input=preds[5][i], size=(h, w), mode='bilinear', align_corners=True)
            loss_up_att.append(self.criterion2(pred_up, targets_up))
        loss_up_att = sum(loss_up_att)

        #decomp lp
        lower_bg_node = 1-one_hot_hb_list[2]
        lower_parts = []
        for i in self.lower_part_list:
            lower_parts.append(one_hot_pb_list[i])
        targets_lp = torch.stack(lower_parts, dim=1)
        targets_lp = targets_lp.argmax(dim=1,keepdim=False)
        targets_lp[lower_bg_node==1]=self.ignore_index
        loss_lp_att = []
        for i in range(len(preds[5])):
            pred_lp = F.interpolate(input=preds[5][i], size=(h, w), mode='bilinear', align_corners=True)
            loss_lp_att.append(self.criterion2(pred_lp, targets_lp))
        loss_lp_att = sum(loss_lp_att)

        # dependency decomposition
        # loss_context_att =[]
        loss_dp_att = []
        for i in range(len(preds[-2])):
            # loss_context = []
            loss_dp = []
            for j in range(self.num_classes-1):
                part_list = self.part_list_list[j]
                parts_onehot = []
                for k in part_list:
                    parts_onehot.append(one_hot_pb_list[k+1])
                parts_bg_node = 1-sum(parts_onehot)
                targets_dp_onehot = torch.stack(parts_onehot, dim=1)
                targets_dp = targets_dp_onehot.argmax(dim=1, keepdim=False)
                targets_dp[parts_bg_node == 1] = self.ignore_index
                pred_dp = F.interpolate(input=preds[-2][i][j], size=(h, w), mode='bilinear', align_corners=True)
                loss_dp.append(self.criterion2(pred_dp, targets_dp))
                
            loss_dp = sum(loss_dp)
            loss_dp_att.append(loss_dp)
        loss_dp_att = sum(loss_dp_att)

        # dsn loss
        pred_dsn = F.interpolate(input=preds[-1], size=(h, w), mode='bilinear', align_corners=True)
        loss_dsn = self.criterion(pred_dsn, targets[0])
        return loss[0] + 0.4*loss_hb[0] + 0.4*loss_fb[0] + (sum(loss[1:]) + 0.4*sum(loss_hb[1:]) + 0.4*sum(loss_fb[1:]))/(len(preds[1])-1)+ 0.4 * loss_dsn + 0.2*(loss_fh_att + loss_up_att + loss_lp_att + loss_dp_att)/len(preds[3])