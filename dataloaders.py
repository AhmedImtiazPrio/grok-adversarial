import torch
import torchvision
from torchvision import transforms


def cifar10_dataloaders(config):
    """
    Vanilla cifar10 dataloader. No augmentations
    """

    torch.manual_seed(config.seed)

    train_transform = transforms.Compose(
    [
          transforms.ToTensor(),
          transforms.ConvertImageDtype(torch.float32),
          transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)
          ) if config.normalize else lambda x:x
    ])

    val_transform = transforms.Compose(
    [
          transforms.ToTensor(),
          transforms.ConvertImageDtype(torch.float32),
          transforms.Normalize((0.4914, 0.4822, 0.4465),(0.2023, 0.1994, 0.2010)
          ) if config.normalize else lambda x:x
    ])

    train = torchvision.datasets.CIFAR10(root='./data', train=True,
      transform=train_transform, download=True)
    train_loader = torch.utils.data.DataLoader(train,
                                            batch_size=config.train_batch_size,
                                            shuffle=True)

    test = torchvision.datasets.CIFAR10(root='./data', train=False,
      transform=val_transform, download=True)
    test_loader = torch.utils.data.DataLoader(test,
                                            batch_size=config.eval_batch_size,
                                            shuffle=True) ## shuffling
                                                          ## to get random
                                                          ## neighborhood for LC
    return train_loader, test_loader


def get_LC_samples(dloader,config):
    """
    Selects a set of samples for LC computation. TODO: allow subsampling classwise
    """

    samples = []
    labels = []

    size = 0
    for x,y in dloader:
        samples.append(x)
        labels.append(y)
        size += x.shape[0]
        if size >= config.approx_n: break

    ## concat and keep LC_batch_size
    samples = torch.concatenate(samples,axis=0)[:config.approx_n]
    labels = torch.concatenate(labels,axis=0)[:config.approx_n]

    return samples, labels