




class Decode:
    eval_file_path  = 'data/ssd_eval.pkl'
    data_dir        = 'data/images'
    variances = [0.1, 0.1, 0.2, 0.2]
    batch_size      = 16    
    checkpoint_dir  = 'checkpoints'
    checkpoint_path = 'ssd_1_6.4616_5.9102.pth'