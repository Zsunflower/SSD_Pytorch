import torch
import torch.nn as nn
import numpy as np
from config import Config
from model import SSDModel
from VGG19BaseSSD import Vgg19BaseSSD
from box_utils import BoxUtils


class SSDDecoder(nn.Module):

    def __init__(self, predictor_shapes, scales, aspect_ratios,
                 img_width, img_height, variances, n_classes, conf_thresh=0.5, iou_thresh=0.01):
        super(SSDDecoder, self).__init__()
        size = np.asarray([img_width, img_height, img_width, img_height])
        anchor_boxes = BoxUtils.generate_anchor_boxes_model(predictor_shapes, scales, aspect_ratios) #(nboxes, 4)
        anchor_boxes = np.expand_dims(anchor_boxes, axis=0) #(1, nboxes, 4)
        anchor_boxes = BoxUtils.corner2center(anchor_boxes) #(cx, cy, w, h)
        self.anchor_template = torch.as_tensor(anchor_boxes, dtype=torch.float)
        self.size = torch.as_tensor(size, dtype=torch.float)
        self.variances = torch.as_tensor(variances, dtype=torch.float)
        self.conf_thresh = conf_thresh
        self.iou_thresh = iou_thresh
        self.n_classes = n_classes


    def forward(self, output):
        #output (batch, nboxes, nclasses + 4)
        output[:, :, [-4, -3, -2, -1]] *= self.variances
        output[:, :, [-4, -3]] = output[:, :, [-4, -3]] * self.anchor_template[:, :, [-2, -1]] + self.anchor_template[:, :, [-4, -3]]
        output[:, :, [-2, -1]] = torch.exp(output[:, :, [-2, -1]]) * self.anchor_template[:, :, [-2, -1]]

        output[:, :, [-4, -3]] -= output[:, :, [-2, -1]] / 2.0
        output[:, :, [-2, -1]] += output[:, :, [-4, -3]]

        output[:, :, -4:] *= self.size
        output[:, :, :-4] = torch.exp(output[:, :, :-4])
        probs_max, class_max = torch.max(output[:, :, :-4], dim=-1)
        class_max = torch.unsqueeze(class_max, dim=-1) #(batch, nboxes, 1)
        probs_max = torch.unsqueeze(probs_max, dim=-1) #(batch, nboxes, 1)
        class_max = class_max.type(output.dtype)
        output = torch.cat([class_max, probs_max, output[:, :, -4:]], dim=-1) #(batch, nboxes, 6) (class_id, conf, xmin, ymin, xmax, ymax)
        return output


if __name__ == '__main__':
    config = Config()
    device = 'cpu'

    if config.eval_cfg.model_name == 'SSDModel':
        model = SSDModel(config.img_width, config.img_height, config.nclasses, config.scales, config.aspect_ratios)
    elif config.eval_cfg.model_name == 'Vgg19BaseSSD':
        model = Vgg19BaseSSD(config.img_width, config.img_height, config.nclasses, config.scales, config.aspect_ratios)

    model = model.to(device)
    precitor_shapes = model.get_predictor_shapes(device)
    image = torch.randn(1, 3, config.img_height, config.img_width).to(device)
    ssd_output = model(image)
    ssd_output = ssd_output.to(device)

    ssd_decoder = SSDDecoder(precitor_shapes, config.scales, config.aspect_ratios,
                             config.img_width, config.img_height, config.variances, config.nclasses, 
                             conf_thresh=config.eval_cfg.threshold, iou_thresh=config.eval_cfg.iou_threshold)
    traced = torch.jit.trace(ssd_decoder, ssd_output)
    traced.save('ssd_decoder.pth')