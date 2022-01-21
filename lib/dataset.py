import torch 
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt


def download_dataset(name, batch_size=32):
    
    if name == 'cifar-10':
        return cifar_10(batch_size)
    
    if name == 'mnist':
        return mnist(batch_size)

def cifar_10(batch_size):
    
    transform = transforms.Compose([
        transforms.ToTensor(), 
        transforms.Normalize((0.5, 0.5, 0.5), (0.5, 0.5, 0.5))
    ])

    trainset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.CIFAR10(root='./data/cifar-10', train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)

    classes = ('plane', 'car', 'bird', 'cat', 'deer', 'dog', 'frog', 'horse', 'ship', 'truck')

    return train_dataloader, test_dataloader, classes


def mnist(batch_size):
    
    transform = transforms.Compose([transforms.ToTensor()])

    trainset = torchvision.datasets.MNIST(root='./data/mnist', train=True, download=True, transform=transform)
    train_dataloader = torch.utils.data.DataLoader(trainset, batch_size=batch_size, shuffle=True, num_workers=1)

    testset = torchvision.datasets.MNIST(root='./data/mnist', train=False, download=True, transform=transform)
    test_dataloader = torch.utils.data.DataLoader(testset, batch_size=batch_size, shuffle=False, num_workers=1)
    
    return train_dataloader, test_dataloader


def show_sample_from_dataset(train_dataloader, classes=None):
    
    batch = next(iter(train_dataloader))
    
    plt.figure(figsize=(10, 2))    
    
    for i in range(5):
        x, y = batch[0][i], batch[1][i].item()
        img = x / 2 + 0.5
        img = img.permute(1, 2, 0).numpy()  
        y = classes[y] if classes else y

        plt.subplot(1, 5, i+1)
        plt.imshow(img)
        plt.title(y)
        plt.axis('off')

    plt.show()
    