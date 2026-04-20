## Deep Networks Always Grok and Here is Why, ICML 2024
[![Open In Colab](https://colab.research.google.com/assets/colab-badge.svg)](https://bit.ly/grok-adv-demo)
[![Open In Colab](https://img.shields.io/badge/Weights_&_Biases-FFCC33?logo=WeightsAndBiases&logoColor=black)](https://bit.ly/grok-adv-trak)


### [Paper Link](https://arxiv.org/abs/2402.15555) | [Website](https://bit.ly/grok-adversarial)
<div style="display:flex;">
<img src="https://imtiazhumayun.github.io/grokking/img/cnn_cifar10_nobn.svg" alt="Cifar10-CNN" style="width: 30%; height: auto; margin-bottom: 10px;">
<img src="https://imtiazhumayun.github.io/grokking/img/resnet_imagenette_nobn.svg" alt="Cifar10-CNN" style="width: 30%; height: auto; margin-bottom: 10px;">
</div>
Fig: CNN grokking CIFAR10 adversarial examples (Left), ResNet18 grokking Imagenette adv examples (Right) 
<br>
<br>
<br>


### Local Complexity Computation

This codebase computes the "local complexity" of a neural network inside a cross-polytopal region of the input space. At a high level:
1.  **Neighborhood Sampling**: For any point in the input space, we sample a neighborhood (hull) of points by adding and subtracting random orthogonal vectors.
2.  **Intersection Counting**: We pass these points through the model and check if the activation pattern (sign of activations) changes for any neuron across the neighborhood. A change indicates that the neuron's decision boundary intersects the neighborhood.
3.  **Complexity Metric**: The number (or percentage) of intersecting neurons is counted as a measure of local complexity.



## Getting Started

### Environment Setup

You can recreate the conda environment used in this project with the provided `grokspline.yml` file:

```bash
conda env create -f grokspline.yml
conda activate grokspline
```

### Training


#### MNIST (MLP)
To train the MLP model on MNIST with default settings, you must provide a log comment:

```bash
python train_mlp_mnist.py "my_experiment_comment"
```

You can override configurations defined in `configs.py` by passing the attribute name and value as additional arguments:

```bash
python train_mlp_mnist.py "my_experiment_comment" "lr" 0.001
```

#### CIFAR10 (ResNet18)
To train the ResNet18 model on CIFAR10 with default settings:

```bash
python train_resnet18_cifar10.py
```

You can override configurations defined in `configs.py` via command line arguments. For example:

```bash
python train_resnet18_cifar10.py --lr 0.001 --use_ffcv False
```

### Citation

```bibtex
@inproceedings{humayun2024grok,
author = {Humayun, Ahmed Imtiaz and Balestriero, Randall and Baraniuk, Richard},
title = {Deep networks always grok and here is why},
year = {2024},
booktitle = {International Conference on Machine Learning},
}
```
