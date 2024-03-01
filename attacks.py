## Attack models. TODO: Add layerwise attacks

import torch

class PGD(object):
    """
    Projected Gradient Descent.
    Implementation from: https://github.com/Harry24k/PGD-pytorch
    """
    def __init__(self, model,
                 eps=8/255, alpha=2/255,
                 steps=100,
                 random_start=True,
                 lmax=None, lmin=None,
                 device='cuda'
                ):
        self.model = model
        self.eps = eps
        self.alpha = alpha
        self.steps = steps
        self.random_start = random_start
        self.device = device
        self.lmin = lmin
        self.lmax = lmax

    def __call__(self,images,labels):
        images = images.to(self.device)
        labels = labels.to(self.device)
        loss = torch.nn.CrossEntropyLoss()

        ori_images = images.data
        lmin = ori_images.min() if self.lmin is None else self.lmin
        lmax = ori_images.max() if self.lmax is None else self.lmax

        for i in range(self.steps) :

            images.requires_grad = True
            outputs = self.model(images)

            self.model.zero_grad()
            cost = loss(outputs, labels).to(self.device)
            cost.backward()

            adv_images = images + self.alpha*images.grad.sign()
            eta = torch.clamp(adv_images - ori_images, min=-self.eps, max=self.eps)
            images = torch.clamp(ori_images + eta, min=lmin, max=lmax).detach_()

        return images