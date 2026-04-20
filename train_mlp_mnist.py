"""
Training script for Multi-Layer Perceptron (MLP) on MNIST dataset.
Computes local complexity during training.

Example usage:
    python train_mlp_mnist.py "my_experiment_comment"

Overriding config attributes:
    python train_mlp_mnist.py "my_experiment_comment" "attr_name" value
    Example: python train_mlp_mnist.py "test_run" "lr" 0.001
"""
from collections import defaultdict
from itertools import islice
import random
import time
from pathlib import Path
import math

import numpy as np
from tqdm.auto import tqdm

import torch
import torch.nn as nn
import torchvision
import torchvision.transforms as transforms

import matplotlib.pyplot as plt

from utils import flatten_model
from local_complexity import get_intersections_for_samples
from samplers import get_ortho_hull_around_samples

from configs import config_base_mnist_mlp

import time
import os
import sys

def cycle(iterable):
    """Yields elements from the iterable indefinitely."""
    while True:
        for x in iterable:
            yield x

def compute_accuracy(network, dataset, device, N=2000, batch_size=50):
    """Computes accuracy of `network` on `dataset`.
    """
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        correct = 0
        total = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            logits = network(x.to(device))
            predicted_labels = torch.argmax(logits, dim=1)
            correct += torch.sum(predicted_labels == labels.to(device))
            total += x.size(0)
        return (correct / total).item()

def compute_loss(network, dataset, loss_function, device, N=2000, batch_size=50):
    """Computes mean loss of `network` on `dataset`.
    """
    with torch.no_grad():
        N = min(len(dataset), N)
        batch_size = min(batch_size, N)
        dataset_loader = torch.utils.data.DataLoader(dataset, batch_size=batch_size, shuffle=True)
        loss_fn = loss_function_dict[loss_function](reduction='sum')
        one_hots = torch.eye(10, 10).to(device)
        total = 0
        points = 0
        for x, labels in islice(dataset_loader, N // batch_size):
            y = network(x.to(device))
            if loss_function == 'CrossEntropy':
                total += loss_fn(y, labels.to(device)).item()
            elif loss_function == 'MSE':
                total += loss_fn(y, one_hots[labels]).item()
            points += len(labels)
        return total / points
    
optimizer_dict = {
    'AdamW': torch.optim.AdamW,
    'Adam': torch.optim.Adam,
    'SGD': torch.optim.SGD
}

activation_dict = {
    'ReLU': nn.ReLU,
    'Tanh': nn.Tanh,
    'Sigmoid': nn.Sigmoid,
    'GeLU': nn.GELU
}

loss_function_dict = {
    'MSE': nn.MSELoss,
    'CrossEntropy': nn.CrossEntropyLoss
}

############ Init Params #############

config = config_base_mnist_mlp()

assert len(sys.argv) > 1, 'Must have log comment'

comment = sys.argv[1] ## will be appended to logname
log_pref = config.log_name + comment 

timestamp = time.ctime().replace(' ','_')
log_dir = os.path.join('./logs/',log_pref+'::'+timestamp)
config.log_dir = log_dir
os.mkdir(log_dir)


model_dir = os.path.join('./models/',log_pref+'::'+timestamp)
config.model_dir = model_dir
os.mkdir(model_dir)

### config setattr
# Allow overriding config attributes from command line: python train_mlp_mnist.py "comment" "attr_name" value
if len(sys.argv) == 4:
    setattr(config,sys.argv[2],sys.argv[3])
    print(f'Setting {sys.argv[2]} as {sys.argv[3]}')

## training params

train_points = int(config.train_points)
optimization_steps = int(config.optimization_steps)
batch_size = int(config.batch_size)
loss_function = config.loss_function
optimizer = config.optimizer     # 'AdamW' or 'Adam' or 'SGD'
weight_decay = float(config.weight_decay)
lr = float(config.lr)

initialization_scale = float(config.initialization_scale)
    
depth = config.depth   
width = config.width
act = config.activation    # 'ReLU' or 'Tanh' or 'Sigmoid' or 'GELU'

log_freq = math.ceil(optimization_steps / 150)

## local complexity approximation parameters
approx_n = int(config.approx_n)
hull_n = config.hull_n
hull_r = config.hull_r
seed = config.seed
approx_batch_size = int(config.approx_batch_size)

## save config
torch.save(config,os.path.join(log_dir,'config.pt'))
print('Training Config:')
print(config.__dict__)

############# Set device and Seeds ######################

device = torch.device('cuda:0' if torch.cuda.is_available() else 'cpu')
dtype = torch.float64

torch.set_default_dtype(dtype)
torch.manual_seed(seed)
torch.cuda.manual_seed_all(seed)
random.seed(seed)
np.random.seed(seed)

###### load dataset #############

# Load MNIST dataset and create random subset if specified
if config.dataset == 'mnist':

    train = torchvision.datasets.MNIST(root='./data', train=True, 
        transform=torchvision.transforms.ToTensor(), download=True) # load train
    train_idx = torch.randint(0,len(train),(train_points,)) # sample train
    train = torch.utils.data.Subset(train, train_idx) 
    train_loader = torch.utils.data.DataLoader(train, batch_size=batch_size, shuffle=True) #loader

    test = torchvision.datasets.MNIST(root='./data', train=False, 
        transform=torchvision.transforms.ToTensor(), download=True) #load test

    # get train and test points for local non-linearity approximation
    train_x = torch.cat([each for each,_ in train_loader])
    test_x = [each for each,_ in torch.utils.data.DataLoader(test,batch_size=len(test), shuffle=False)][0]
    rand_x = torch.rand(approx_n,*train_x.shape[1:]) ### [0,1]^784
    
    ## get random labels for train
    
    if config.label_noise:
        
        print('Randomizing Labels...')
        
        train_y = torch.randint(0,10,(train_points,))
        train = torch.utils.data.TensorDataset(
                train_x,
                train_y)

        train_loader = torch.utils.data.DataLoader(
            train,
            batch_size=batch_size,
            shuffle=True   
        )
    
else:
    raise NotImplementedError("Only MNIST dataset is supported.")
    
################ Init loops ########################


if type(hull_r) != list:
    hull_r = [hull_r]

if type(width) != list:
    width = [width]
    
if type(depth) != list:
    depth = [depth]

    
for _width,_depth in zip(width,depth):
    
    print(f'Training width {_width} and depth {_depth}')

    ############# Model Definition ######################
    
    if config.model == 'mlp':
    
        assert act in activation_dict, f"Unsupported activation function: {act}"
        activation_fn = activation_dict[act]

        # Create Multi-Layer Perceptron (MLP) model
        layers = [nn.Flatten()]
        for i in range(_depth):
            if i == 0:
                layers.append(nn.Linear(np.prod(config._in_dims), _width))
                layers.append(activation_fn())
            elif i == _depth - 1:
                layers.append(nn.Linear(_width, config._n_classes))
            else:
                layers.append(nn.Linear(_width, _width))
                layers.append(activation_fn())
        mlp = nn.Sequential(*layers).to(device)
        mlp.cuda();

        with torch.no_grad():
            for p in mlp.parameters():
                p.data = initialization_scale * p.data


        torch.save(
                    mlp,
                    os.path.join(
                        model_dir,
                        f'checkpoint-r:nohook-w:{_width}-d:{_depth}-s:{-1}.pt'
                    )
                )

        ############### Initialize Hooks ###################
        # Hooks are used to capture intermediate activations, 
        # which are needed to compute local complexity.
      
        names,modules = flatten_model(mlp)
        assert len(names) == len(modules)

        target_ids = np.asarray([i for i,each in enumerate(modules) if type(each)==torch.nn.modules.Linear])

        global activation

        activation = {}
        def get_activation(name):
            def hook(model, input, output):
                activation[name] = output.detach()
            return hook

        for each in target_ids:
            modules[each].register_forward_hook(get_activation(names[each]))

        layer_names = np.sort(np.asarray(names)[target_ids])
    
    else:
        raise NotImplementedError("Only MLP model is supported.")

    ################ train ###################

    # create optimizer
    assert optimizer in optimizer_dict, f"Unsupported optimizer choice: {optimizer}"
    opt = optimizer_dict[optimizer](mlp.parameters(), lr=lr, weight_decay=weight_decay)
    
    if config.load_model is not None:
        state_dict = torch.load(config.load_model)
        opt.load_state_dict(state_dict['optimizer_state_dict'])
        mlp.load_state_dict(state_dict['model_state_dict'])
    
    # define loss function
    assert loss_function in loss_function_dict
    loss_fn = loss_function_dict[loss_function]()


    train_losses = []
    test_losses = []
    train_accuracies = []
    test_accuracies = []
    norms = []
    last_layer_norms = []
    log_steps = []
    density_approx_train = dict(zip(hull_r,[[] for i in range(len(hull_r))]))
    density_approx_test = dict(zip(hull_r,[[] for i in range(len(hull_r))]))
    density_approx_random = dict(zip(hull_r,[[] for i in range(len(hull_r))]))

    steps = 0

    one_hots = torch.eye(10, 10).to(device)
    with tqdm(total=optimization_steps) as pbar:
        for x, labels in islice(cycle(train_loader), optimization_steps):

            # Periodic logging and checkpointing
            if (steps < 30) \
              or (steps < 150 and steps % 10 == 0) \
              or (steps < 1000 and steps % 100 == 0) \
              or (steps < 10000 and steps % 1000 == 0) \
              or steps % log_freq == 0:
                    
                mlp.eval()
                
                ### checkpoint before anything
                torch.save(
                    {
                        'model_state_dict': mlp.state_dict(),
                        'optimizer_state_dict': opt.state_dict(),
                    },
                    os.path.join(
                        model_dir,
                        f'checkpoint-w:{_width}-d:{_depth}-s:{steps}.pt'
                    )
                )
                
                for r in hull_r:
                    
                    ### approx local nonlinearity
                    n_inters, p_inters = get_intersections_for_samples(
                                          train_x,
                                          model=mlp,
                                          batch_size=min(approx_batch_size,train_x.shape[0]),
                                          sampler=get_ortho_hull_around_samples,
                                          sampler_params={'n' : hull_n, 'r' : r, 'seed':seed},
                                          layer_names=layer_names,
                                          activation_buffer=activation
                                         )


                    density_approx_train[r].append(n_inters)

                    n_inters, p_inters = get_intersections_for_samples(
                                      test_x,
                                      model=mlp,
                                      batch_size=min(approx_batch_size,test_x.shape[0]),
                                      sampler=get_ortho_hull_around_samples,
                                      sampler_params={'n' : hull_n, 'r' : r, 'seed':seed},
                                      layer_names=layer_names,
                                      activation_buffer=activation
                                                         )


                    density_approx_test[r].append(n_inters)
                    

                    n_inters, p_inters = get_intersections_for_samples(
                                      rand_x,
                                      model=mlp,
                                      batch_size=min(approx_batch_size,rand_x.shape[0]),
                                      sampler=get_ortho_hull_around_samples,
                                      sampler_params={'n' : hull_n, 'r' : r, 'seed':seed},
                                      layer_names=layer_names,
                                      activation_buffer=activation
                                                         )


                    density_approx_random[r].append(n_inters)
                    

                log_steps.append(steps)
                
                ### compute losses
                train_losses.append(compute_loss(mlp, train, loss_function, device, N=len(train)))
                train_accuracies.append(compute_accuracy(mlp, train, device, N=len(train)))
                test_losses.append(compute_loss(mlp, test, loss_function, device, N=len(test)))
                test_accuracies.append(compute_accuracy(mlp, test, device, N=len(test)))
                
                ### calculate norm

                    
                pbar.set_description("L: {0:1.1e}|{1:1.1e}. A: {2:2.1f}%|{3:2.1f}%".format(
                    train_losses[-1],
                    test_losses[-1],
                    train_accuracies[-1] * 100, 
                    test_accuracies[-1] * 100))
            
                
                mlp.train()
                
                ## end logging
                
            ## gradient update 

            opt.zero_grad()
            y = mlp(x.to(device))
            if loss_function == 'CrossEntropy':
                loss = loss_fn(y, labels.to(device))
            elif loss_function == 'MSE':
                loss = loss_fn(y, one_hots[labels])
            loss.backward()
            opt.step()

            steps += 1
            pbar.update(1)


    torch.save(
        {
            'layers': layer_names,
            'r': hull_r,
            'n': hull_n,
            'width':_width,
            'depth':_depth,
            'config': config,
            'comment': comment,
            'train_idx': train_idx,
            'train_losses': np.asarray(train_losses),
            'test_losses': np.asarray(test_losses),
            'train_accuracies': np.asarray(train_accuracies),
            'test_accuracies': np.asarray(test_accuracies),

            'log_steps': np.asarray(log_steps),
            'density_approx_train': density_approx_train,
            'density_approx_test': density_approx_test,
            'density_approx_random': density_approx_random,

        },


        os.path.join(
            log_dir,f'w:{_width}-d:{_depth}-seed:{seed}.pt')
    )
    
    log_steps = np.asarray(log_steps)
    log_steps += 1
        
    
    for r in hull_r:
        
        
        ax = plt.subplot(1, 1, 1)
        ax.plot(log_steps, train_accuracies, color='red', label='train')
        ax.plot(log_steps, test_accuracies, color='green', label='test')
        plt.xscale('log')
        plt.xlim(1, None)
        plt.xlabel("Optimization Steps")
        plt.ylabel("Accuracy")
        plt.legend(loc='lower left')
        
        ax2 = ax.twinx()
        ax2.set_ylabel("Local Complexity")
        ax2.plot(log_steps,
                 torch.stack(density_approx_train[r]).sum(2).mean(1),
                 color='purple', label='train')
        ax2.plot(log_steps,
                 torch.stack(density_approx_test[r]).sum(2).mean(1),
                 color='orange', label='test')
        ax2.plot(log_steps,
                 torch.stack(density_approx_random[r]).sum(2).mean(1),
                 color='blue', label='rand')

        plt.legend(loc='upper left')
        plt.title(f"depth-{_depth} width-{_width} {act} MLP \nα = {initialization_scale} radius = {r}, hull samples = {hull_n}", fontsize=11)

        plt.savefig(
            os.path.join(
                log_dir,f'r:{r}-w:{_width}-d:{_depth}-seed:{seed}.png'
            ),
            dpi=600, bbox_inches='tight',pad_inches=0)
        
        plt.close()
        
        del ax
        
    del mlp












