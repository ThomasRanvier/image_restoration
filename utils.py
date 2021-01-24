import datasets

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
        'blury_lfw': datasets.load_blury_lfw,
    }
    return datasets_switch.get(dataset_name, lambda x: _raise(f'Dataset {dataset_name} unknown!'))(batch_size)