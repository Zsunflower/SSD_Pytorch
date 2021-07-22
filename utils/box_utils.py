import numpy as np


class BoxUtils:
    @staticmethod
    def iou_vector(boxes1, boxes2):
        # Caculate iou between each pair (boxes1[i], boxes2[i])
        # boxes1: numpy array (N, 4) coordinates of boxes 1 (xmin, ymin, xmax, ymax)
        # boxes2: numpy array (N, 4) coordinates of boxes 2 (xmin, ymin, xmax, ymax)
        # return: numpy array (N, ) iou of earch pair in boxes1 and boxes2
        x1 = np.maximum(boxes1[:, 0], boxes2[:, 0])
        y1 = np.maximum(boxes1[:, 1], boxes2[:, 1])
        x2 = np.minimum(boxes1[:, 2], boxes2[:, 2])
        y2 = np.minimum(boxes1[:, 3], boxes2[:, 3])
        widths = x2 - x1
        heights = y2 - y1
        widths[widths < 0] = 0
        heights[heights < 0] = 0
        intersection_area = widths * heights
        a_area = (boxes1[:, 2] - boxes1[:, 0]) * (boxes1[:, 3] - boxes1[:, 1])
        b_area = (boxes2[:, 2] - boxes2[:, 0]) * (boxes2[:, 3] - boxes2[:, 1])
        iou = intersection_area / (a_area + b_area - intersection_area)
        return iou

    @staticmethod
    def iou_matrix(boxes1, boxes2):
        # Caculate iou between each pair (boxes1[i], boxes2[j]) 0<i<m, 0<j<n
        # boxes1: numpy array (M, 4) coordinates of boxes1 (xmin, ymin, xmax, ymax)
        # boxes2: numpy array (N, 4) coordinates of boxes2 (xmin, ymin, xmax, ymax)
        # return: numpy array (M, N) iou of earch pair in boxes1[i] and boxes2[j]
        ious = []
        boxes1 = np.expand_dims(boxes1, axis=1)  # (M, 1, 4)
        boxes1 = np.tile(boxes1, (1, len(boxes2), 1))  # (M, N, 4)
        for b in boxes1:
            iou = BoxUtils.iou_vector(b, boxes2)
            ious.append(iou)
        return np.asarray(ious)

    @staticmethod
    def generate_anchor_boxes(predictor_shape, scales, aspect_ratios):
        # Generate anchor boxes for predictor shape predictor_shape
        # predictor_shape:(height, width) shape of predictor
        # scales: Ex [0.25, 0.5, 0.75] scales of anchor boxes
        # aspect_ratios: Ex [1, 2, 0.5] ratios of anchor boxes
        # return: numpy array shape (n_boxes, 4)
        # anchor boxes(format xmin, ymin, xmax, ymax)

        _scales = np.tile(
            np.expand_dims(np.asarray(scales), axis=-1), (1, len(aspect_ratios))
        ).flatten()
        _aspect_ratios = np.tile(np.asarray(aspect_ratios), (len(scales)))

        widths = _scales * np.sqrt(_aspect_ratios)
        heights = _scales / np.sqrt(_aspect_ratios)
        x_center = (np.arange(predictor_shape[1]) + 0.5) / predictor_shape[1]
        y_center = (np.arange(predictor_shape[0]) + 0.5) / predictor_shape[0]

        x_center, y_center = np.meshgrid(x_center, y_center)
        widths, x_center = np.meshgrid(widths, x_center)
        heights, y_center = np.meshgrid(heights, y_center)
        x_min = x_center - 0.5 * widths
        y_min = y_center - 0.5 * heights
        x_max = x_center + 0.5 * widths
        y_max = y_center + 0.5 * heights
        anchor_boxes = np.dstack([x_min, y_min, x_max, y_max])
        return anchor_boxes.reshape((-1, 4))

    @staticmethod
    def corner2center(batch_corner, in_place=True):
        # batch_corner: (b, n, 4) format (xmin, ymin, xmax, ymax)
        if not in_place:
            batch_corner = batch_corner.copy()
        # (xmin, ymin, xmax, ymax) -> (xmin, ymin, width, height)
        batch_corner[:, :, [-2, -1]] -= batch_corner[:, :, [-4, -3]]
        # (xmin, ymin, width, height) -> (cx, cy, width, height)
        batch_corner[:, :, [-4, -3]] += batch_corner[:, :, [-2, -1]] / 2.0
        return batch_corner

    @staticmethod
    def center2corner(batch_center, in_place=True):
        # batch_center: (b, n, 4) format(cx, cy, width, height)
        if not in_place:
            batch_center = batch_center.copy()
        # (cx, cy, width, height) -> (xmin, ymin, width, height)
        batch_center[:, :, [-4, -3]] -= batch_center[:, :, [-2, -1]] / 2.0
        # (xmin, ymin, width, height) -> (xmin, ymin, xmax, ymax)
        batch_center[:, :, [-2, -1]] += batch_center[:, :, [-4, -3]]
        return batch_center

    @staticmethod
    def generate_anchor_boxes_model(predictor_shapes, scales, aspect_ratios):
        # Generate list anchor boxes for each predictor layer

        anchor_boxes = []
        for (predictor_shape, scale) in zip(predictor_shapes, scales):
            anchor_boxes_predictor = BoxUtils.generate_anchor_boxes(
                predictor_shape, [scale], aspect_ratios
            )
            anchor_boxes.append(anchor_boxes_predictor)
        anchor_boxes = np.concatenate(anchor_boxes, axis=0)  # (nboxes, 4)
        return anchor_boxes
