import numpy as np
import torch
from box_utils import BoxUtils



class SSDLabelEncoder():
    #Encode ground truth boxes to format match output of the model
    def __init__(self, anchor_boxes_template, nclasses, img_height, img_width,
                 pos_iou_threshold=0.5, neg_iou_threshold=0.3, variance=[0.1, 0.1, 0.2, 0.2]):
        #anchor_boxes_template: (nboxes, 4) anchor boxes output by the model
        #nclasses: number of positive classes
        nclasses += 1 #One for background class
        self.nclasses = nclasses
        self.img_height = img_height
        self.img_width  = img_width
        self.pos_iou_threshold = pos_iou_threshold
        self.neg_iou_threshold = neg_iou_threshold
        self.variance = variance
        # self.anchor_boxes_template = anchor_boxes_template
        class_one_hot = np.zeros(shape=(len(anchor_boxes_template), nclasses)) #shape (nboxes, nclasses)
        class_one_hot[:, 0] = 1 #Default all anchor boxes are background
        output_template = np.concatenate([class_one_hot, anchor_boxes_template], axis=-1) #shape (nboxes, nclasses + 4)
        self.output_template = np.expand_dims(output_template, axis=0) #shape (1, nboxes, nclasses + 4)

        self.class_one_hots = np.eye(self.nclasses) #one hot vector for each class (nclasses, nclasses)

    def bipartite_match(self, similarities_):
        #Match earch ground truth box with anchor box that max IOU
        #similarities: (M, N) similarities matrix of ground truth boxes(M) and anchor boxes(N)
        #return:(M, ) anchor box indices that matched to ground truth boxes
        similarities = similarities_.copy()
        matches = np.zeros(len(similarities), dtype=np.int)
        sim_indices = range(len(similarities))
        for i in range(len(similarities)):
            ac_indicies_max = np.argmax(similarities, axis=1)
            ac_values_max = similarities[sim_indices, ac_indicies_max]
            gt_indicies_max = np.argmax(ac_values_max)
            matches[gt_indicies_max] = ac_indicies_max[gt_indicies_max]
            #Remove row and column matched
            similarities[gt_indicies_max, :] = 0
            similarities[:, matches[gt_indicies_max]] = 0
        return matches

    def multi_match(self, similarities_):
        #Match earch anchor box(not matched) with ground truth box that IOU > self.pos_iou_threshold
        #similarities: (M, N) similarities matrix of ground truth boxes(M) and anchor boxes(N)
        #return: 2 (L, ) ground truth box indices and anchor box indices that matched
        similarities = similarities_.copy()
        ac_indices = range(similarities.shape[1])
        #Match each anchor box to ground truth box that max IOU
        gt_indices_max = np.argmax(similarities, axis=0) #(N, )
        #Get IOUs that matched
        gt_values_max = similarities[gt_indices_max, ac_indices] #(N, )
        #Create mask IOUs > pos_iou_threshold
        mask = gt_values_max > self.pos_iou_threshold
        #Get anchor box index that matched to mask
        ac_indices_thresh_matched = np.nonzero(mask)[0]
        #Get ground truth box that matched to mask
        gt_indices_thresh_matched = gt_indices_max[ac_indices_thresh_matched]
        return gt_indices_thresh_matched, ac_indices_thresh_matched

    def match(self, ground_truth_boxes):
        #Assign ground truth boxes to anchor boxes
        #ground_truth_boxes: list length batch_size of array (M, 5) groundtruth boxes (class_id, xmin, ymin, xmax, ymax)
                                # (M number of ground truth boxes of each batch item)    (0,      , 1   , 2   , 3   , 4)
        #return: anchor boxes labeled
        batch_size = len(ground_truth_boxes)
        output = np.tile(self.output_template, (batch_size, 1, 1)) #(batch_size, nboxes, nclasses + 4)
        # output = self.generate_output_template(batch_size)
        for i in range(batch_size):
            label_i = ground_truth_boxes[i].astype(np.float) #(M, 5)
            if len(label_i) < 1:
                continue
            #Normalize bboxes to range [0, 1]
            label_i[:, [1, 3]] /= self.img_width
            label_i[:, [2, 4]] /= self.img_height
            #Get one hot vectors of label category
            class_one_hot = self.class_one_hots[label_i[:, 0].astype(np.int)] #shape (M, nclasses)
            label_one_hot = np.concatenate([class_one_hot, label_i[:, 1: ]], axis=-1) #shape (M, nclasses + 4)
            #Caculate IOU matrix between ground truth and anchor boxes
            similarities = BoxUtils.iou_matrix(label_i[:, 1: ], output[i, :, -4:])
            maches = self.bipartite_match(similarities)
            output[i, maches, :] = label_one_hot
            similarities[:, maches] = 0 #Erase columns that matched
            #Do multi matching
            gt_matched, ac_matched = self.multi_match(similarities)
            output[i, ac_matched, :] = label_one_hot[gt_matched]
            similarities[:, ac_matched] = 0
            #Set all remain anchor boxes that IOU > neg_iou_threshold to be ignored(not treat as background)
            max_iou_background = np.amax(similarities, axis=0)
            ignored_anchor_boxes = np.nonzero(max_iou_background >= self.neg_iou_threshold)[0]
            output[i, ignored_anchor_boxes, 0] = 0
        return output

    def rescale(self, output):
        #caculate offsets for regression the loss function
        #output: (batch, nboxes, nclasses + 4) encoded output(after matched to ground truth boxes)
        #        (xmin, ymin, xmax, ymax)
        #        (-4,   -3,   -2,   -1)
        #Convert (xmin, ymin, xmax, ymax) to (cx, cy, w, h)
        output[:, :, -4:] = BoxUtils.corner2center(output[:, :, -4:])

        batch_size = output.shape[0]

        anchor_template = np.tile(self.output_template, (batch_size, 1, 1)) #(batch_size, nboxes, nclasses + 4)
        anchor_template[:, :, -4:] = BoxUtils.corner2center(anchor_template[:, :, -4:])

        output[:, :, [-4, -3]] -= anchor_template[:, :, [-4, -3]] #XY(ground truth) - XY(anchor box)
        output[:, :, [-4, -3]] /= anchor_template[:, :, [-2, -1]] #(XY(ground truth) - XY(anchor box)) / WH(anchor box)

        output[:, :, [-2, -1]] = np.log(output[:, :, [-2, -1]] / anchor_template[:, :, [-2, -1]])
        output[:, :, -4:] /= self.variance
        return output

    def __call__(self, ground_truth_boxes):
        output_matched = self.match(ground_truth_boxes)
        output_rescaled = self.rescale(output_matched)
        return torch.as_tensor(output_rescaled, dtype=torch.float)