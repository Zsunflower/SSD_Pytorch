

class TrainConfig:
    train_file_path = 'data/ssd_train.pkl'
    eval_file_path  = 'data/ssd_eval.pkl'
    data_dir        = 'data/images'
    batch_size      = 16
    neg_pos_ratio   = 3
    alpha           = 1
    num_epochs      = 60
    checkpoint_dir  = 'checkpoints'
    checkpoint_file = 'ssd_1_6.4616_5.9102.pth'
    
    
class EvalConfig:
    eval_file_path  = 'data/ssd_eval.pkl'
    data_dir        = 'data/images'
    batch_size      = 16    
    checkpoint_dir  = 'checkpoints'
    checkpoint_file = 'ssd_58_1.0070_1.3091.pth'
    groundtruths    = 'eval/groundtruths'
    detections      = 'eval/detections'
    results         = 'eval/results'
    debug_imgs      = 'eval/visual_detection'
    threshold       = 0.5
    iou_threshold   = 0.01
    cmd_path        = '/content/Object-Detection-Metrics/pascalvoc.py'


class Config:
    
    img_width       = 512
    img_height      = 512
    nclasses        = 1
    scales          = [0.05167422, 0.08641827, 0.11201055, 0.15300986]
    aspect_ratios   = [2.15237637, 2.97160099, 3.49238618, 3.97957222, 4.54607464]
    variances       = [0.1, 0.1, 0.2, 0.2]
    
    train_cfg = TrainConfig()
    eval_cfg  = EvalConfig()