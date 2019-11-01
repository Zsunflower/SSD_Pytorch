import numpy as np
from box_utils import BoxUtils




def decode_output(output, anchor_template, variances, img_width, img_height, n_classes,
                  conf_thresh=0.5, iou_thresh=0.01):
    #output: output of model (batch_size, nboxes, nclasses + 4)
    #conf_thresh: minimum confidence for treat as a object
    #iou_thresh: every boxes with IOU > iou_thresh will be filtered
    output = output.copy()
    batch_size = output.shape[0]

    anchor_boxes = np.expand_dims(anchor_template, axis=0) #(1, nboxes, 4)
    anchor_boxes = np.tile(anchor_boxes, (batch_size, 1, 1))
    anchor_boxes = BoxUtils.corner2center(anchor_boxes) #(cx, cy, w, h)

    output[:, :, [-4, -3, -2, -1]] *= variances
    output[:, :, [-4, -3]] = output[:, :, [-4, -3]] * anchor_boxes[:, :, [-2, -1]] + anchor_boxes[:, :, [-4, -3]]
    output[:, :, [-2, -1]] = np.exp(output[:, :, [-2, -1]]) * anchor_boxes[:, :, [-2, -1]]

    output[:, :, -4:] = BoxUtils.center2corner(output[:, :, -4:])
    output[:, :, -4:] *= [img_width, img_height, img_width, img_height]
    output[:, :, :-4] = np.exp(output[:, :, :-4])

    class_max = np.argmax(output[:, :, :-4], axis=-1) #(batch, nboxes)
    class_max = np.expand_dims(class_max, axis=-1) #(batch, nboxes, 1)
    probs_max = np.max(output[:, :, :-4], axis=-1) #batch, nboxes
    probs_max = np.expand_dims(probs_max, axis=-1) #(batch, nboxes, 1)
    output = np.concatenate([class_max, probs_max, output[:, :, -4:]], axis=-1) #(batch, nboxes, 6) (class_id, conf, xmin, ymin, xmax, ymax)
    batch_output = []
    for output_item in output:
        boxes_filtered = []
        #output_item: (nboxes, 6)
        #filter background
        output_item = output_item[output_item[:, 0] > 0]
        output_item = output_item[output_item[:, 1] > conf_thresh]
        for class_id in range(1, n_classes + 1):
            #get all boxes with class_id
            box_class_id = output_item[output_item[:, 0] == class_id]
            #Perform nms on class id
            nms_filtered = nms(box_class_id, iou_thresh)
            if len(nms_filtered) > 0:
                boxes_filtered.extend(nms_filtered)
            batch_output.append(boxes_filtered)
    return batch_output


def nms(boxes, iou_thresh):
    #boxes: (nboxes, 6) (class_id, conf, xmin, ymin, xmax, ymax)
    boxes = boxes.copy()
    boxes_left = []
    while len(boxes) > 0:
        #Find box that max conf
        max_index = np.argmax(boxes[:, 1]) #Get box with highest conf
        boxes_left.append(boxes[max_index])
        boxes = np.delete(boxes, max_index, axis=0)
        if len(boxes) < 1:
            break
        similarities = BoxUtils.iou_matrix(boxes[:, 2:], np.expand_dims(boxes_left[-1][2:], axis=0)) #(N, 1)
        similarities = np.squeeze(similarities, axis=-1)
        boxes = boxes[similarities <= iou_thresh]
    return boxes_left