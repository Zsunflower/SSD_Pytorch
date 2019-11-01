import torch
import cv2
import os
import shutil
from config import Config
from model import SSDModel
from data_utils import SSDDataset, SSDDataAugmentation, Transpose, Normalization, collate_sample
from torchvision import transforms
from decode_utils import decode_output

device = 'cuda' if torch.cuda.is_available() else 'cpu'

class Eval:
    
    def __init__(self, cfg):
        self.cfg = cfg
        self.parse_config()
    
    def build_model(self):
        self.model = SSDModel(self.cfg.img_width, self.cfg.img_height, self.cfg.nclasses, self.cfg.scales, self.cfg.aspect_ratios).to(device)
        checkpoint_path = os.path.join(self.cfg.eval_cfg.checkpoint_dir, self.cfg.eval_cfg.checkpoint_file)
        if not os.path.exists(checkpoint_path):
            print("Can't load checkpoint from: ", checkpoint_path)
        else:
            print("Eval model from checkpoint: ", checkpoint_path)
            self.model.load_state_dict(torch.load(checkpoint_path))

    
    def prepare_data(self):
        eval_aug = SSDDataAugmentation(target_size={'h': self.cfg.img_height, 'w': self.cfg.img_width}, train=False)  
        eval_ds  = SSDDataset(self.cfg.eval_cfg.data_dir,
                              self.cfg.eval_cfg.eval_file_path,
                              transform=transforms.Compose([eval_aug,
                                                            Transpose(),
                                                            Normalization(127.5, 127.5)
                                                            ]))        
        self.eval_loader  = torch.utils.data.DataLoader(dataset=eval_ds,  batch_size=self.cfg.eval_cfg.batch_size,
                                                        collate_fn=collate_sample, shuffle=False)
    
    def parse_config(self):
        self.build_model()
        self.prepare_data()
        if os.path.exists(self.cfg.eval_cfg.groundtruths):
            shutil.rmtree(self.cfg.eval_cfg.groundtruths)
        if os.path.exists(self.cfg.eval_cfg.detections):
            shutil.rmtree(self.cfg.eval_cfg.detections)
        if os.path.exists(self.cfg.eval_cfg.results):
            shutil.rmtree(self.cfg.eval_cfg.results)
        if os.path.exists(self.cfg.eval_cfg.debug_imgs):
            shutil.rmtree(self.cfg.eval_cfg.debug_imgs)
        os.makedirs(self.cfg.eval_cfg.groundtruths, exist_ok=True)
        os.makedirs(self.cfg.eval_cfg.detections, exist_ok=True)
        os.makedirs(self.cfg.eval_cfg.results, exist_ok=True)
        os.makedirs(self.cfg.eval_cfg.debug_imgs, exist_ok=True)


    def run(self):
        step = 0
        total = 0
        font = cv2.FONT_HERSHEY_SIMPLEX        
        for sample in self.eval_loader:
            batch_images, batch_labels, batch_filenames = sample['image'], sample['objs'], sample['filename']
            batch_images = batch_images.to(device)
            total += len(batch_images)
            y_pred = self.model(batch_images)
            y_pred = y_pred.cpu().data.numpy()
            y_pred_decoded = decode_output(y_pred, self.model.generate_anchor_boxes(device), self.cfg.variances, 
                                           conf_thresh=self.cfg.eval_cfg.threshold, iou_thresh=self.cfg.eval_cfg.iou_threshold)

            for (yp, label, filename) in zip(y_pred_decoded, batch_labels, batch_filenames):
                img = cv2.imread(os.path.join(images_dir, filename))
                h, w = img.shape[: 2]
                scaley, scalex = h / img_height, w / img_width

                f = open(os.path.join(self.cfg.eval_cfg.groundtruths, os.path.basename(filename).split('.')[0] + '.txt'), 'w')
                for box in label:
                    f.write("{} {} {} {} {}\n".format(int(box[0]), box[1], box[2], box[3], box[4]))
                    box[1: ] *= [scalex, scaley, scalex, scaley]
                    box = box.astype(np.int)
                    img = cv2.rectangle(img, (box[1], box[2]), (box[3], box[4]), (0, 0, 255), 2)
                f.close()

                f = open(os.path.join(self.cfg.eval_cfg.detections, os.path.basename(filename).split('.')[0] + '.txt'), 'w')
                for box in yp:
                    f.write("{0} {1:.4f} {2} {3} {4} {5}\n".format(int(box[0]), box[1], int(box[2]), int(box[3]), int(box[4]), int(box[5])))
                    box[2: ] *= [scalex, scaley, scalex, scaley]
                    b = box[2: ].astype(np.int)
                    img = cv2.rectangle(img, (b[0], b[1]), (b[2], b[3]), (255, 0, 0), 2)
                    cv2.putText(img, '{:.4f}'.format(box[1]), (b[0], b[1]), font, 1, (0, 0, 255), 2)
                f.close()
                cv2.imwrite(os.path.join(self.cfg.eval_cfg.debug_imgs, os.path.basename(filename)), img)
        print("Eval done!")


if __name__ == '__main__':
    config = Config()
    eval   = Eval(config)
    eval.run()