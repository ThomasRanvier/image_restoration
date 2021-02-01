import argparse
import models
import utils
from enum import Enum


available_models = {
    'dense_autoencoder': models.DenseAutoencoder,
    'conv_autoencoder': models.ConvAutoencoder,
    'u_net': models.UNet,
}

available_datasets = [
    'mnist',
    'noisy_mnist',
    'fashion_mnist',
    'noisy_fashion_mnist',
    'cifar',
    'noisy_cifar',
    'lfw',
    'blurry_lfw',
]

input_shapes = {
    'mnist': (28, 28),
    'noisy_mnist': (28, 28),
    'fashion_mnist': (28, 28),
    'noisy_fashion_mnist': (28, 28),
    'cifar': (32, 32, 3),
    'noisy_cifar': (32, 32, 3),
    'lfw': (128, 128, 3),
    'blurry_lfw': (128, 128, 3),
}

    
if __name__ == "__main__":
    parser = argparse.ArgumentParser()
    parser.add_argument('-m', '--model-name', type=str, choices=available_models.keys(),
                        default='dense_autoencoder', help='name of the model to train')
    parser.add_argument('-d', '--dataset-name', type=str, choices=available_datasets,
                        default='mnist', help='name of the dataset to use')
    parser.add_argument('-e', '--epochs', type=int,
                        default=20, help='number of epochs, default is 20')
    parser.add_argument('-b', '--batch-size', type=int,
                        default=128, help='batch size, default is 128')
    args = parser.parse_args()
    
    (train_ds, eval_ds, _) = utils.load_dataset(args.dataset_name, batch_size=args.batch_size)
    model = available_models[args.model_name](input_shapes[args.dataset_name])
    eval_losses = model.fit(train_ds, eval_ds, epochs=args.epochs)
    model.save_weights(f'trained_models/{args.model_name}_d_{args.dataset_name}_b_{args.batch_size}_e_{args.epochs}')
    utils.plot_losses(eval_losses, f'plots/{args.model_name}_d_{args.dataset_name}_b_{args.batch_size}_e_{args.epochs}.png')
    