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
    # dataset = load_dataset('data/ADE20K/24_classes.npy', 'data/ADE20K/images.npy')
    # dataset = load_dataset('data/lhq_256/24_classes_rgb.npy', 'data/lhq_256/images.npy')
    dataset = load_dataset('data/ADE20K/24_classes_rgb.npy', 'data/ADE20K/images.npy')

    args = {
        'input_size': (256, 256, 3),
        'output_size': (256, 256, 3),
        'd_p_lr': 2e-4,
        'd_g_lr': 2e-4,
        'gan_lr':  4e-4,
        'gan_loss_weights': [1, 1, 10],
        'main_log_path': "logs",
        'g_path_save': "generators",
        'd_path_save': None,  # "discriminators",
        'evaluate_path_save': "images",
        'model_code_save': 'code',
        'save_with_optimizer': False,
        'logging': True,
        'evaluate_per_step': 100,
    }

        
    print(dataset[0].dtype, dataset[1].dtype, dataset[1].max(), dataset[1].min())
    print(dataset[0].shape, dataset[1].shape)

    
    # dataset = dataset[0][:5000], dataset[1][:5000]
    gan = GAN_Training(**args)
    gan.train(200, dataset, save_per_epochs=1, batch_size=10, categorical_input=False, random_rot90=True)