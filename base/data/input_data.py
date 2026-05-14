from bindsnet.datasets import MNIST
from bindsnet.encoding import PoissonEncoder
from torchvision import transforms

import torch


import os


class Data:
    def __init__(self, time, dt, shuffle = True, intensity=64) :
        self.time = time
        self.dt = dt
        self.shuffle = shuffle
        self.root = "../../data"
        self.intensity = intensity

        self.encoder = PoissonEncoder(time=self.time, dt=self.dt)
        self.train_dataset = None
        self.test_dataset = None
        
    def load_MNIST(self):
        self.train_dataset = MNIST(self.encoder, root=os.path.join(self.root, "MNIST"), download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * self.intensity)]), train=True)
        self.test_dataset = MNIST(self.encoder, root=os.path.join(self.root, "MNIST"), download=True, transform = transforms.Compose([transforms.ToTensor(), transforms.Lambda(lambda x: x * self.intensity)]), train=False)
        return self.train_dataset, self.test_dataset
    