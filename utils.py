import datasets

def _raise(ex):
    raise NotImplementedError(ex)

def load_dataset(dataset_name='mnist', batch_size=128):
    datasets_switch = {
        'mnist': datasets.load_mnist,
        'noisy_mnist': datasets.load_noisy_mnist,
        'fashion_mnist': datasets.load_fashion_mnist,
        'noisy_fashion_mnist': datasets.load_noisy_fashion_mnist,
    }
    return datasets_switch.get(dataset_name, lambda x: _raise(f'Dataset {dataset_name} unknown!'))(batch_size)