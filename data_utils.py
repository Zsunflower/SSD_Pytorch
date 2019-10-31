import numpy as np
import pandas as pd
import cv2
import os
from imgaug import augmenters as iaa
from imgaug.augmentables.bbs import BoundingBoxesOnImage



class SSDDataset(torch.utils.data.Dataset):

    def __init__(self, root, pkl_file, transform=None):
        self.df = pd.read_pickle(pkl_file)
        self.root = root
        self.transform = transform

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):
        filename = self.df.iloc[idx, 0]
        objs     = self.df.iloc[idx, 1]
        objs     = np.asarray(objs, dtype=np.float)
        full_path = os.path.join(self.root, filename)
        image    = cv2.imread(full_path)
        sample   = {'image': image, 'objs': objs, 'filename': filename}
        if self.transform:
            sample = self.transform(sample)
        return sample


class Transpose(object):
    """Change image format from (h, w, c) to (c, h, w)."""

    def __call__(self, sample):
        image, objs, filename = sample['image'], sample['objs'], sample['filename']
        # swap color axis
        # numpy image: H x W x C to C x H x W
        image = np.transpose(image, (2, 0, 1))
        return {'image': image, 'objs': objs, 'filename': filename}
    
    
    
class Normalization(object):
    """Rescale the image pixels to a specific range."""
    
    def __init__(self, mean, std):
        self.mean = mean
        self.std  = std
    
    def __call__(self, sample):
        image, objs, filename = sample['image'], sample['objs'], sample['filename']
        image = image.astype(np.float)
        image = (image - self.mean) / self.std
        return {'image': image, 'objs': objs, 'filename': filename}
    
    
    
class SSDDataAugmentation(object):

    def __init__(self, target_size={'h': 512, 'w': 512},
                 random_brightness={'low': -32, 'high': 32, 'prob': 0.5},
                 random_contrast={'low': 0.5, 'high': 1.5, 'prob': 0.5},
                 random_saturation={'low': 0.5, 'high': 2.5, 'prob': 0.5},
                 random_hue={'low': 0.5, 'high': 1.5, 'prob': 0.5},
                 channel_shuffle={'prob': 0.5},
                 random_translate={'low': -0.2, 'high': 0.2},
                 random_scale={'min': 0.7, 'max': 1.2},
                 prob=0.8,
                 train=True):
        self.train = train
        if self.train:
            self.seq_train = iaa.Sequential([iaa.Sometimes(random_brightness['prob'], iaa.Add((random_brightness['low'], random_brightness['high']))),
                                    iaa.Sometimes(random_contrast['prob'], iaa.LinearContrast((random_contrast['low'], random_contrast['high']))),
                                    iaa.Sometimes(random_saturation['prob'], iaa.MultiplySaturation((random_saturation['low'], random_saturation['high']))),
                                    iaa.Sometimes(random_hue['prob'], iaa.MultiplyHue((random_hue['low'], random_hue['high']))),
                                    iaa.ChannelShuffle(channel_shuffle['prob']),
                                    iaa.Sometimes(0.5, iaa.Affine(translate_percent={'x': (random_translate['low'], random_translate['high']), 
                                                                    'y': (random_translate['low'], random_translate['high'])},
                                                scale={'x': (random_scale['min'], random_scale['max']),
                                                        'y': (random_scale['min'], random_scale['max'])})),
                                    iaa.Resize({"height": target_size['h'], "width": target_size['w']})])
            self.prob = prob
        self.seq_eval = iaa.Resize({"height": target_size['h'], "width": target_size['w']})


    def __call__(self, sample):
        image, objs, filename = sample['image'], sample['objs'], sample['filename']
        class_objs = np.expand_dims(objs[:, 0], axis=1) #(N, 1)
        bbs = BoundingBoxesOnImage.from_xyxy_array(objs[:, 1:], shape=image.shape)
        if self.train:
            image, bbs_aug = self.seq_train(image=image, bounding_boxes=bbs)
        else:
            image, bbs_aug = self.seq_eval(image=image, bounding_boxes=bbs)
        bbs_aug = bbs_aug.to_xyxy_array()
        objs = np.concatenate([class_objs, bbs_aug], axis=1)
        return {'image': image, 'objs': objs, 'filename': filename}

    
    
def collate_sample(list_samples):
    #list_samples: list of dictionary {'image': image, 'objs': objs}
    #return: dictionary {'image': np.asarray(list all images), 'objs': list of all objs}
    image_batched = np.asarray([sample['image'] for sample in list_samples])
    objs_batched  = [sample['objs'] for sample in list_samples]
    filename_batched = [sample['filename'] for sample in list_samples]
    return {'image': torch.as_tensor(image_batched, dtype=torch.float, device=device),
            'objs': objs_batched,
            'filename': filename_batched}
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    
    