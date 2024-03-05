import torch
import torchvision
from torchvision import transforms
from typing import List

import os
import time
import warnings


CIFAR_MEAN = [125.307, 122.961, 113.8575]
CIFAR_STD = [51.5865, 50.847, 51.255]

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
                                            batch_size=config.test_batch_size,
                                            shuffle=True) ## shuffling
                                                          ## to get random
                                                          ## neighborhood for LC
    return train_loader, test_loader


def cifar10_dataloaders_ffcv(config,
                             train_path='./data/cifar10_train.beton',
                             test_path='./data/cifar10_test.beton',
                             precision='fp32',
                             os_cache=True,
                             num_workers=2
                            ):
    """
    Create ffcv dataloaders if ffcv is available
    """
    
    try:
        from ffcv.fields import IntField, RGBImageField
        from ffcv.fields.decoders import IntDecoder, SimpleRGBImageDecoder
        from ffcv.fields.rgb_image import CenterCropRGBImageDecoder, \
            RandomResizedCropRGBImageDecoder
        from ffcv.loader import Loader, OrderOption
        from ffcv.pipeline.operation import Operation
        from ffcv.transforms import RandomHorizontalFlip, Cutout, \
            RandomTranslate, Convert, ToDevice, ToTensor, ToTorchImage
        from ffcv.transforms.common import Squeeze
        from ffcv.writer import DatasetWriter
        
    except:
        warnings.warn("cant import ffcv. falling back to legacy dataloader")
        return cifar10_dataloaders(config)
    
    
    paths = {
        'train': train_path,
        'test': test_path
    }
    
    ### create ffcv datasets if not exists
    if not os.path.exists(train_path):

        print(f'{train_path} not found. Creating FFCV dataset...')
    
        datasets = {
                'train': torchvision.datasets.CIFAR10('./data', train=True, download=True),
                'test': torchvision.datasets.CIFAR10('./data', train=False, download=True)
                }
        
        for (name, ds) in datasets.items():
        
            path = paths[name]

            writer = DatasetWriter(path, {
                'image': RGBImageField(),
                'label': IntField()
            })
            writer.from_indexed_dataset(ds)
    
    loaders = {}

    for name in ['train', 'test']:
        label_pipeline: List[Operation] = [IntDecoder(), ToTensor(),
         ToDevice(torch.device('cuda:0')), Squeeze()]
        image_pipeline: List[Operation] = [SimpleRGBImageDecoder()]

        ### add augmentations for train
        if name == 'train' and config.use_aug:
            print('Using training augmentations')

            image_pipeline.extend([
                RandomHorizontalFlip(),
                RandomTranslate(padding=2, fill=tuple(map(int, CIFAR_MEAN))),
                Cutout(4, tuple(map(int, CIFAR_MEAN))),
            ])

        image_pipeline.extend([
            ToTensor(),
            ToDevice(torch.device('cuda:0'), non_blocking=True),
            ToTorchImage(),
            Convert(torch.float16) if precision == 'fp16' else Convert(torch.float32),
        ])
        
        if config.normalize:
            image_pipeline.extend([
                torchvision.transforms.Normalize(CIFAR_MEAN, CIFAR_STD)
            ])
            
        
        ordering = OrderOption.RANDOM # if name == 'train' else OrderOption.SEQUENTIAL

        loaders[name] = Loader(paths[name],
                               batch_size=getattr(config,f'{name}_batch_size'),
                               num_workers=num_workers,
                               order=ordering, drop_last=(name == 'train'), os_cache=os_cache,
                               pipelines={'image': image_pipeline, 'label': label_pipeline})
    
    return loaders['train'], loaders['test']


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