import torch
import torch.nn as nn
import numpy as np
from box_utils import BoxUtils


class SSDModel(nn.Module):

    def __init__(self, width, height, n_classes, scales, aspect_ratios):
        #(height, width): size of input images
        #n_classes: Number of positive classes
        #scales: list of scales per predictor layer
        #aspect_ratios: list of ratios
        super(SSDModel, self).__init__()
        n_classes   = n_classes + 1
        self.scales = scales
        self.aspect_ratios = aspect_ratios
        self.n_classes = n_classes
        self.n_boxes   = len(self.aspect_ratios)
        self.width  = width
        self.height = height

        self.block1 = self.create_block(3 , 32, (5, 5), (2, 2))
        self.mp1    = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.block2 = self.create_block(32, 48, (3, 3), (1, 1))
        self.mp2    = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.block3 = self.create_block(48, 64, (3, 3), (1, 1))
        self.mp3    = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.block4 = self.create_block(64, 64, (3, 3), (1, 1))
        self.mp4    = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.block5 = self.create_block(64, 48, (3, 3), (1, 1))
        self.mp5    = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.block6 = self.create_block(48, 48, (3, 3), (1, 1))
        self.mp6    = nn.MaxPool2d(kernel_size=(2, 2), stride=(2, 2))
        self.block7 = self.create_block(48, 32, (3, 3), (1, 1))

        self.class4 = self.create_classes_predict_block(64)
        self.class5 = self.create_classes_predict_block(48)
        self.class6 = self.create_classes_predict_block(48)
        self.class7 = self.create_classes_predict_block(32)
        self.class_logsoftmax = nn.LogSoftmax(dim=2)

        self.boxes4 = self.create_boxes_predict_block(64)
        self.boxes5 = self.create_boxes_predict_block(48)
        self.boxes6 = self.create_boxes_predict_block(48)
        self.boxes7 = self.create_boxes_predict_block(32)

    def create_block(self, in_channels, out_channels, 
                     kernel_size, padding, bn_momentum=0.99):
        block = nn.Sequential(nn.Conv2d(in_channels=in_channels, out_channels=out_channels,
                                        kernel_size=kernel_size, stride=(1, 1), padding=padding),
                              nn.BatchNorm2d(out_channels, momentum=bn_momentum),
                              nn.ELU())
        return block

    def create_classes_predict_block(self, in_channels):
        return nn.Conv2d(in_channels=in_channels, out_channels=self.n_classes * self.n_boxes, 
                         kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def create_boxes_predict_block(self, in_channels):
        return nn.Conv2d(in_channels=in_channels, out_channels=self.n_boxes * 4,
                         kernel_size=(3, 3), stride=(1, 1), padding=(1, 1))

    def forward(self, x):
        y = self.block1(x)
        y = self.mp1(y)
        y = self.block2(y)
        y = self.mp2(y)
        y = self.block3(y)
        y = self.mp3(y)
        y = self.block4(y)
        class4 = self.class4(y).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.n_classes) #(batch, n4, nclasses)
        boxes4 = self.boxes4(y).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)              #(batch, n4, 4)

        y = self.mp4(y)
        y = self.block5(y)
        class5 = self.class5(y).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.n_classes) #(batch, n5, nclasses)
        boxes5 = self.boxes5(y).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)              #(batch, n5, 4)

        y = self.mp5(y)
        y = self.block6(y)
        class6 = self.class6(y).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.n_classes) #(batch, n6, nclasses)
        boxes6 = self.boxes6(y).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)              #(batch, n6, 4)

        y = self.mp6(y)
        y = self.block7(y)
        class7 = self.class7(y).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, self.n_classes) #(batch, n7, nclasses)
        boxes7 = self.boxes7(y).permute(0, 2, 3, 1).contiguous().view(x.size(0), -1, 4)              #(batch, n7, 4)

        class_concated = torch.cat([class4, class5, class6, class7], dim=1) #shape (batch, n4+n5+n6+n7, nclasses)
        class_concated = self.class_logsoftmax(class_concated)              #shape (batch, n4+n5+n6+n7, nclasses)
        boxes_concated = torch.cat([boxes4, boxes5, boxes6, boxes7], dim=1) #shape (batch, n4+n5+n6+n7, 4)
        prediction = torch.cat([class_concated, boxes_concated], dim=2)     #shape (batch, n4+n5+n6+n7, nclasses + 4)
        return prediction

    def get_predictor_shapes(self, device):
        #Return shape of each predictor layers(4, 5, 6, 7)
        anchor_box_shapes = []
        x = torch.randn(1, 3, self.height, self.width, device=device)
        y = self.block1(x)
        y = self.mp1(y)
        y = self.block2(y)
        y = self.mp2(y)
        y = self.block3(y)
        y = self.mp3(y)
        y = self.block4(y)
        anchor_box_shapes.append((y.size(2), y.size(3)))
        y = self.mp4(y)
        y = self.block5(y)
        anchor_box_shapes.append((y.size(2), y.size(3)))
        y = self.mp5(y)
        y = self.block6(y)
        anchor_box_shapes.append((y.size(2), y.size(3)))
        y = self.mp6(y)
        y = self.block7(y)
        anchor_box_shapes.append((y.size(2), y.size(3)))
        return anchor_box_shapes

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