"""
Training GAN

@author: Adrian Kucharski
"""
import json
import os
from dataset import DataIterator, load_dataset
from model_double_d import GAN_Training

os.environ['TF_CPP_MIN_LOG_LEVEL'] = '3'
# os.environ["CUDA_VISIBLE_DEVICES"] = "-1"


if __name__ == '__main__':
    dataset = load_dataset('data/lhq_256/24_classes.npy', 'data/lhq_256/images.npy')


    args = {
        'input_size': (256, 256, 25),
        'd_p_lr': 4e-4,
        'd_g_lr': 4e-4,
        'gan_lr': 2e-4,
        'gan_loss_weights': [1, 1, 2],
        'main_log_path': "logs",
        'g_path_save': None, #"generators",
        'd_path_save': None, #"discriminators",
        'evaluate_path_save': "images",
        'log_path':  "tf_logs",
        'model_code_save': 'code',
        'save_with_optimizer': False,
        'logging': True
    }

    gan = GAN_Training(**args)
    
    dataset = (dataset[0][:10000], dataset[1][:10000])
    gan.train(50, dataset, save_per_epochs=1, batch_size=8)
