import numpy as np
import torch.nn as nn
from utils.pc_noise_layers.identity import Identity


class Noiser(nn.Module):
    def __init__(self, noise_layers: list):
        super(Noiser, self).__init__()
        # self.noise_layers = [Identity()]
        if noise_layers == None:
            self.noise_layers = [Identity()]
        else:
            self.noise_layers = []
            for layer in noise_layers:
                self.noise_layers.append(layer)

    def forward(self, pc):
        random_noise_layer = np.random.choice(self.noise_layers, 1)[0]
        # print(random_noise_layer)
        return random_noise_layer(pc)

