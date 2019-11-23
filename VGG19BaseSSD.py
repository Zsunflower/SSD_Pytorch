import torch
import torch.nn as nn
import torchvision
import numpy as np
from box_utils import BoxUtils


class Vgg19BaseSSD(nn.Module):

    def __init__(self,  width, height, n_classes, scales, aspect_ratios):
        super(Vgg19BaseSSD, self).__init__()
        n_classes   = n_classes + 1
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.n_classes = n_classes
        self.n_boxes   = len(self.aspect_ratios)
        self.width  = width
        self.height = height
        self.class_logsoftmax = nn.LogSoftmax(dim=2)

        vgg19 = torchvision.models.vgg19(pretrained=True)
        self.vgg19_base = vgg19.features
        self.predict_layers_indices = [20, 25, 34]
        self.in_channels            = [512, 512, 512]
        class1 = self.create_classes_predict_block(512)
        class2 = self.create_classes_predict_block(512)
        class3 = self.create_classes_predict_block(512)
        boxes1 = self.create_boxes_predict_block(512)
        boxes2 = self.create_boxes_predict_block(512)
        boxes3 = self.create_boxes_predict_block(512)
        self.classes_predict_layer = nn.ModuleList([class1, class2, class3])
        self.boxes_predict_layer   = nn.ModuleList([boxes1, boxes2, boxes3])

    def create_classes_predict_block(self, in_channels):
        return nn.Conv2d(in_channels=in_channels, out_channels=self.n_classes * self.n_boxes, 
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def create_boxes_predict_block(self, in_channels):
        return nn.Conv2d(in_channels=in_channels, out_channels=self.n_boxes * 4,
                        kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def freeze_base(self):
        for param in self.vgg19_base.parameters():
            param.requires_grad = False

    def unfreeze_base(self):
        for param in self.vgg19_base.parameters():
            param.requires_grad = True

    def forward(self, x):
        predict_layers = []
        classes = []
        boxes   = []
        for i, layer in enumerate(self.vgg19_base):
            x = layer(x)
            if i in self.predict_layers_indices:
                predict_layers.append(x)
        for cls_layer, box_layer, layer in zip(self.classes_predict_layer, self.boxes_predict_layer, predict_layers):
            classes.append(cls_layer(layer).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.n_classes))
            boxes.append(box_layer(layer).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4))
        classes_concated = torch.cat(classes, dim=1)
        classes_concated = self.class_logsoftmax(classes_concated)
        boxes_concated   = torch.cat(boxes,   dim=1)
        output = torch.cat([classes_concated, boxes_concated], dim=2)
        return output

    def get_predictor_shapes(self, device):
        #Return shape of each predictor layers(4, 5, 6, 7)
        anchor_box_shapes = []
        x = torch.randn(1, 3, self.height, self.width, device=device)
        for i, layer in enumerate(self.vgg19_base):
            x = layer(x)
            if i in self.predict_layers_indices:
                anchor_box_shapes.append((x.size(2), x.size(3)))
        return anchor_box_shapes
            
    @torch.jit.export
    def generate_anchor_boxes(self, device):
        #Generate list anchor boxes for each predictor layer
        #
        predictor_shapes = self.get_predictor_shapes(device)

        anchor_boxes = []
        for (predictor_shape, scale) in zip(predictor_shapes, self.scales):
            anchor_boxes_predictor = BoxUtils.generate_anchor_boxes(predictor_shape, [scale], self.aspect_ratios)
            anchor_boxes.append(anchor_boxes_predictor)
        anchor_boxes = np.concatenate(anchor_boxes, axis=0) #(nboxes, 4)
        return anchor_boxes