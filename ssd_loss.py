import torch
import torch.nn as nn


class SSDLoss(nn.Module):

    def __init__(self, alpha=1.0, neg_pos_ratio=3.0):
        super(SSDLoss, self).__init__()
        self.alpha = alpha
        self.neg_pos_ratio = neg_pos_ratio
        self.smooth_l1_loss = nn.SmoothL1Loss(reduction='sum')

    def ssd_conf_loss(self, y_pred, y_true, pos_mask, neg_mask):
        #Caculate confidence loss
        #y_true: (batch, nboxes, nclasses + 4) label of the batch
        #y_pred: (batch, nboxes, nclasses + 4) output of the model
        #pos_mask : (batch, nboxes) positives mask
        #neg_mask : (batch, nboxes) negatives mask
        y_true_pos_boxes = y_true[pos_mask][:, : -4] #shape (n_positive_boxes, nclasses)
        y_pred_pos_boxes = y_pred[pos_mask][:, : -4] #shape (n_positive_boxes, nclasses)
        y_true_neg_boxes = y_true[neg_mask][:, : -4] #shape (n_negative_boxes, nclasses)
        y_pred_neg_boxes = y_pred[neg_mask][:, : -4] #shape (n_negative_boxes, nclasses)
        pos_conf_loss = torch.sum(-y_true_pos_boxes * y_pred_pos_boxes, dim=-1) #shape (n_positive_boxes,)
        neg_conf_loss = torch.sum(-y_true_neg_boxes * y_pred_neg_boxes, dim=-1) #shape (n_negative_boxes,)

        n_neg = int(self.neg_pos_ratio * pos_conf_loss.size(0))
        topk_neg_conf_loss_indices = torch.topk(neg_conf_loss, n_neg)[1]
        topk_neg_loss = neg_conf_loss[topk_neg_conf_loss_indices] #shape (k,)
        return pos_conf_loss.sum() + topk_neg_loss.sum()

    def ssd_loc_loss(self, y_pred, y_true, pos_mask, neg_mask):
        #Caculate localization loss
        #y_true: (batch, nboxes, nclasses + 4) label of the batch
        #y_pred: (batch, nboxes, nclasses + 4) output of the model
        #pos_mask : (batch, nboxes) positives mask
        y_true_pos_boxes = y_true[pos_mask][:, -4:] #shape (n_positive_boxes, 4)
        y_pred_pos_boxes = y_pred[pos_mask][:, -4:] #shape (n_positive_boxes, 4)
        loc_loss = self.smooth_l1_loss(y_pred_pos_boxes, y_true_pos_boxes) #scalar
        return loc_loss

    def forward(self, y_pred, y_true):
        #Caculate loss function
        #y_true: (batch, nboxes, nclasses + 4) label of the batch
        #y_pred: (batch, nboxes, nclasses + 4) output of the model
        pos_mask = torch.max(y_true[:, :, 1: -4], dim=-1)[0] #shape (batch, nboxes)
        pos_mask = pos_mask.bool() #shape (batch, nboxes)
        neg_mask = y_true[:, :, 0] #shape (batch, nboxes)
        neg_mask = neg_mask.bool() #shape (batch, nboxes)
        n_pos_boxes = max(pos_mask.sum(), 1)
        conf_loss = self.ssd_conf_loss(y_pred, y_true, pos_mask, neg_mask)
        loc_loss  = self.alpha * self.ssd_loc_loss(y_pred, y_true, pos_mask, neg_mask)
        return (conf_loss + loc_loss) / n_pos_boxes