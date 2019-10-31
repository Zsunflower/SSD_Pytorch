

class Config:
    img_width = 512
    img_height = 512
    nclasses = 1
    scales = [0.05167422, 0.08641827, 0.11201055, 0.15300986]
    aspect_ratios = [2.15237637, 2.97160099, 3.49238618, 3.97957222, 4.54607464]
    variances = [0.1, 0.1, 0.2, 0.2]
    train_file_path = 'data/ssd_train.pkl'
    eval_file_path  = 'data/ssd_eval.pkl'
    data_dir        = 'data/images'
    batch_size      = 16
    neg_pos_ratio   = 3
    alpha           = 1
    num_epochs      = 60
    checkpoint_dir  = 'checkpoints'
    checkpoint_path = ''
    