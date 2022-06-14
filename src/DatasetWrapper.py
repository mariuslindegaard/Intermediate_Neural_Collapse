import torch
import torch.nn.functional
from torchvision import datasets, transforms  # , models
from torch.utils.data import DataLoader  # , Subset

from typing import Optional, Dict

# TODO(marius): Verify whether shuffling of data is needed


class DatasetWrapper:
    train_loader: DataLoader
    test_loader: DataLoader
    input_batch_shape: torch.Size
    target_batch_shape: torch.Size
    data_id: str
    is_one_hot: bool
    num_classes: int
    batch_size: int
    num_workers: int = 1

    data_download_dir = 'datasets'

    def __init__(self, data_cfg: dict, *args, **kwargs):
        """Init the dataset with given id"""
        self.data_id = data_cfg['dataset-id']
        self.batch_size = data_cfg['batch-size']
        self.num_workers = data_cfg.get('num_workers', 1)

        id_mapping = {
            'cifar10': DatasetWrapper.cifar10,
            'mnist': DatasetWrapper.mnist
        }

        if not self.data_id.lower() in id_mapping.keys():
            raise NotImplementedError(f"Dataset with id '{self.data_id}' is not implemented. "
                                      f"Id must be one of \n{id_mapping.keys()}")

        # Prepeare datset
        train_data, test_data = id_mapping[self.data_id](self, data_cfg, *args, **kwargs)

        self.train_loader = DataLoader(train_data, batch_size=self.batch_size, num_workers=self.num_workers, shuffle=True)
        self.test_loader = DataLoader(test_data, batch_size=self.batch_size, num_workers=self.num_workers)

        tmp_inputs, tmp_targets = next(iter(self.train_loader))
        self.input_batch_shape = tmp_inputs.size()
        self.target_batch_shape = tmp_targets.size()


    def cifar10(self, data_cfg: Optional[Dict] = None, download=True):
        """Cifar10 dataset"""

        normalize = transforms.Normalize(mean=[x/255.0 for x in [125.3, 123.0, 113.9]],
                                         std=[x/255.0 for x in [63.0, 62.1, 66.7]])
        train_tx = transforms.Compose([
            transforms.RandomCrop(32, padding=4),
            transforms.RandomHorizontalFlip(),
            transforms.RandomRotation(15),
            transforms.ToTensor(),
            normalize
        ])
        test_tx = transforms.Compose([
            transforms.ToTensor(),
            normalize
        ])

        train_data = datasets.CIFAR10(root=self.data_download_dir, train=True, download=download,
                                      transform=train_tx)
        test_data = datasets.CIFAR10(root=self.data_download_dir, train=False, download=download,
                                     transform=test_tx)
        self.is_one_hot = False
        self.num_classes = 10
        # self.input_shape = (32, 32, 3)

        return train_data, test_data

    def mnist(self, data_cfg: Optional[Dict] = None, download=True):
        """Mnist dataset"""

        im_size = 28
        padded_im_size = 32

        tx = transforms.Compose([transforms.Pad((padded_im_size - im_size) // 2), transforms.ToTensor(),
                                 transforms.Normalize(mean=[0.1307], std=[0.3081])])

        train_data = datasets.MNIST(root=self.data_download_dir, train=True, download=download,
                                    transform=tx)
        test_data = datasets.MNIST(root=self.data_download_dir, train=False, download=download,
                                   transform=tx)

        self.num_classes = 10
        self.is_one_hot = False
        # self.input_shape = (32, 32, 1)

        return train_data, test_data


def load_dataset_from_dict(data_cfg: dict, *args, **kwargs) -> DatasetWrapper:
    """Load the dataset"""
    wrapper = DatasetWrapper(data_cfg, *args, **kwargs)
    return wrapper
    # (im_size, _, im_channels) = wrapper.input_shape

    # return wrapper.train_loader, im_channels, im_size, wrapper.num_classes


def load_dataset(dataset_id: str, batch_size: int, *args, **kwargs) -> DatasetWrapper:
    """Load the dataset"""
    return load_dataset_from_dict({'dataset-id': dataset_id, 'batch-size': batch_size})


def _test():
    return load_dataset_from_dict({'dataset-id': 'cifar10', 'batch-size': 128})


if __name__ == "__main__":
    _test()
