import torch
import torch.nn as nn
from data_utils import SSDDataset, Transpose, Normalization, SSDDataAugmentation



class Trainer():
    
    def __init__(self, cfg):
        self.cfg = cfg
    
    def train(self):
        pass
    
    def evaluate(self):
        pass
    

    def run(self):
        
