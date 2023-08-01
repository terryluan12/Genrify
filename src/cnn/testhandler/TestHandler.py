import torch
import torchvision
import torchvision.transforms as transforms

class TestHandler:
    def __init__(self, batch_size, num_workers=0):
        """
        TestHandler class to create data loaders for the test data

        Args:
            batch_size (int): batch size
            num_workers (int): number of workers for data loading, default=0
        """
        self.batch_size = batch_size
        self.num_workers = num_workers

        self.spec_test_loader, self.mel_test_loader, self.chroma_test_loader, self.mfcc_test_loader = self._create_test_loaders()

    def _create_test_loaders(self):
        # define the data transformations
        data_transform = transforms.Compose([
                transforms.ToTensor(),
                transforms.Normalize((0.5, 0.5, 0.5),
                                    (0.5, 0.5, 0.5))
            ])
        spec_dataset = torchvision.datasets.ImageFolder('/content/Genrify/src/datasources/spectrogram', transform=data_transform)
        mel_dataset = torchvision.datasets.ImageFolder('/content/Genrify/src/datasources/mel', transform=data_transform)
        chroma_dataset = torchvision.datasets.ImageFolder('/content/Genrify/src/datasources/chroma', transform=data_transform)
        mfcc_dataset = torchvision.datasets.ImageFolder('/content/Genrify/src/datasources/mfcc', transform=data_transform)

        spec_test_loader = torch.utils.data.DataLoader(spec_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        mel_test_loader = torch.utils.data.DataLoader(mel_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        chroma_test_loader = torch.utils.data.DataLoader(chroma_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)
        mfcc_test_loader = torch.utils.data.DataLoader(mfcc_dataset, batch_size=self.batch_size, shuffle=False, num_workers=self.num_workers)

        return spec_test_loader, mel_test_loader, chroma_test_loader, mfcc_test_loader