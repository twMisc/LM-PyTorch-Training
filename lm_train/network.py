"""Classes for constructing the network."""
import torch
from collections import OrderedDict
from torch.func import jacrev


class DNN(torch.nn.Module):
    """Deep neural network class."""

    def __init__(self, layers):
        super(DNN, self).__init__()

        # parameters
        self.depth = len(layers) - 1

        # set up layer order dict
        self.activation = torch.nn.Sigmoid

        layer_list = list()
        layer_list.append(('layer_0', torch.nn.Linear(layers[0], layers[1])))
        layer_list.append(('activation_%d' % 0, self.activation()))
        for i in range(self.depth - 2):
            i = i + 1
            layer_list.append(
                ('layer_%d' % i, torch.nn.Linear(layers[i], layers[i + 1])))
            layer_list.append(('activation_%d' % i, self.activation()))

        layer_list.append(('layer_%d' % (self.depth - 1),
                           torch.nn.Linear(layers[-2], layers[-1],
                                           bias=False)))
        layerDict = OrderedDict(layer_list)

        # deploy layers
        self.layers = torch.nn.Sequential(layerDict)

    def forward(self, x):
        out = self.layers(x)
        return out


class DNN_comb(torch.nn.Module):

    def generate_layers(self, layers, name=''):
        depth = len(layers) - 1
        layer_list = list()
        layer_list.append(
            (f'{name}_layer_{0}', torch.nn.Linear(layers[0], layers[1])))
        layer_list.append((f'{name}_activation_{0}', self.activation()))
        for i in range(depth - 2):
            i = i + 1
            layer_list.append(
                (f'{name}_layer_{i}', torch.nn.Linear(layers[i],
                                                      layers[i + 1])))
            layer_list.append((f'{name}_activation_{i}', self.activation()))

        layer_list.append((f'{name}_layer_{depth-1}',
                           torch.nn.Linear(layers[-2], layers[-1],
                                           bias=False)))
        layerDict = OrderedDict(layer_list)

        # deploy layers
        layers = torch.nn.Sequential(layerDict)
        return layers

    def __init__(self, *layers, s_dim=1):
        super(DNN_comb, self).__init__()
        self.activation = torch.nn.Sigmoid
        self.u_nets = torch.nn.ModuleList()
        for i, layer in enumerate(layers[:-1]):
            self.u_nets.append(self.generate_layers(layer, name=f'u_{i}'))
        self.s_layers = self.generate_layers(layers[-1], name='s')
        self.s_dim = s_dim

    def forward(self, x):
        u_out = [u_net(x) for u_net in self.u_nets]

        s_out = self.s_layers(x[..., self.s_dim])
        return torch.cat([*u_out, s_out], dim=-1)
