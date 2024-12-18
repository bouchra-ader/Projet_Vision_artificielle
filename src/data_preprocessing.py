from torchvision import datasets, transforms
from torch.utils.data import DataLoader

class DataPreprocessor:
    def __init__(self, train_dir, validation_dir, img_size=(150, 150), batch_size=32):
        self.train_dir = train_dir
        self.validation_dir = validation_dir
        self.img_size = img_size
        self.batch_size = batch_size

    def get_loaders(self):
        transform = transforms.Compose([
            transforms.Resize(self.img_size),
            transforms.ToTensor(),
            transforms.Normalize(mean=[0.485, 0.456, 0.406], std=[0.229, 0.224, 0.225]),
        ])

        train_dataset = datasets.ImageFolder(root=self.train_dir, transform=transform)
        validation_dataset = datasets.ImageFolder(root=self.validation_dir, transform=transform)

        train_loader = DataLoader(train_dataset, batch_size=self.batch_size, shuffle=True)
        validation_loader = DataLoader(validation_dataset, batch_size=self.batch_size, shuffle=False)

        return train_loader, validation_loader
