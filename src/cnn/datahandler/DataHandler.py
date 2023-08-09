import torch
import torchvision
import torchvision.transforms as transforms

class DataHandler:
    def __init__(self, data_dir, batch_size, train_ratio=0.75, val_ratio=0.25, num_workers=0):
        """
        DataHandler class to handle the data loading, preprocessing, and creates data loaders

        Args:
            data_dir (str): path to the data directory
            batch_size (int): batch size
            train_ratio (float): ratio of training data, default=0.75
            val_ratio (float): ratio of validation data, default=0.25
            num_workers (int): number of workers for data loading, default=0
        """
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.train_ratio = train_ratio
        self.val_ratio = val_ratio
        self.num_workers = num_workers

        self.train_loader, self.val_loader = self._create_data_loaders()

    def _create_data_loaders(self):
        # define the data transformations
        data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))
            ])
        dataset = torchvision.datasets.ImageFolder(self.data_dir, transform=data_transform)
        num_samples = len(dataset)

        indeces_training  = [x for x in range(int(num_samples*self.train_ratio))]
        indeces_val = [x for x in range(int(num_samples*self.train_ratio), num_samples)]

        train_dataset = torch.utils.data.Subset(dataset, indeces_training)
        val_dataset = torch.utils.data.Subset(dataset, indeces_val)

        train_loader = torch.utils.data.DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        val_loader = torch.utils.data.DataLoader(val_dataset, batch_size=self.batch_size, shuffle=True, num_workers=self.num_workers)
        
        return train_loader, val_loader