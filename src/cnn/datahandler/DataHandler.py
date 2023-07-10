import torch
import torchvision
import torchvision.transforms as transforms

class DataHandler:
    def __init__(self, data_dir, batch_size, train_ratio=0.7, val_ratio=0.15, num_workers=0):
        """
        DataHandler class to handle the data loading, preprocessing, and creates data loaders

        Args:
            data_dir (str): path to the data directory
            batch_size (int): batch size
            train_ratio (float): ratio of training data, default=0.7
            val_ratio (float): ratio of validation data, default=0.15
            num_workers (int): number of workers for data loading, default=0
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers

        self.train_loader, self.val_loader, self.test_loader = self._create_data_loaders()

    def _create_data_loaders(self):
        # define the data transformations
        data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))
            ])
        dataset = torchvision.datasets.ImageFolder(self.data_dir, transform=data_transform)
        num_samples = len(dataset)

        train_size = int(num_samples * self.train_ratio)
        val_size = int(num_samples * self.val_ratio)
        test_size = num_samples - train_size - val_size

        train_dataset, val_dataset, test_dataset = torch.utils.data.random_split(dataset, [train_size, val_size, test_size])

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return train_loader, val_loader, test_loader