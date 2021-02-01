import datasets
import matplotlib.pyplot as plt


def _raise(ex):
    raise NotImplementedError(ex)


def load_dataset(dataset_name, batch_size=128):
    datasets_switch = {
        'mnist': datasets.load_mnist,
        'noisy_mnist': datasets.load_noisy_mnist,
        'fashion_mnist': datasets.load_fashion_mnist,
        'noisy_fashion_mnist': datasets.load_noisy_fashion_mnist,
        'cifar': datasets.load_cifar,
        'noisy_cifar': datasets.load_noisy_cifar,
        'lfw': datasets.load_lfw,
        'blurry_lfw': datasets.load_blurry_lfw,
    }
    return datasets_switch.get(dataset_name, lambda x: _raise(f'Dataset {dataset_name} unknown!'))(batch_size)


def plot_losses(losses, path, title='Evaluation losses'):
    fig = plt.figure()
    plt.plot([i + 1 for i in range(len(losses))], losses, 'red')
    plt.title(title)
    plt.xlabel('epochs')
    plt.ylabel('loss')
    plt.tight_layout()
    plt.savefig(path)