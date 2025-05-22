import torch
import torch.nn as nn
import torch.nn.functional as F
import numpy as np


class PosEnc(nn.Module):
    def __init__(self, C, ks):
        super().__init__()
        self.weight = nn.Parameter(torch.randn(1, C, ks, ks))

    def forward(self, x):
        return  x + self.weight
        
NormLayers = [nn.BatchNorm2d, nn.LayerNorm]


def get_cell_ind(param_name, layers=1):
    if param_name.find('cells.') >= 0:
        pos1 = len('cells.')
        pos2 = pos1 + param_name[pos1:].find('.')
        cell_ind = int(param_name[pos1: pos2])
    elif param_name.startswith('classifier') or param_name.startswith('auxiliary'):
        cell_ind = layers - 1
    elif layers == 1 or param_name.startswith('stem') or param_name.startswith('pos_enc'):
        cell_ind = 0
    else:
        cell_ind = None

    return cell_ind



class MlpNetwork(nn.Module):

    def __init__(self,
                 fc_layers=0,
                 inp_dim = 0,
                 out_dim = 0,
                 ):
        super(MlpNetwork, self).__init__()
    #     print(f"Initializing MlpNetwork with layers: {fc_layers}, input_dim: {inp_dim}, output_dim: {out_dim}", flush=True)

        self.expected_input_sz = inp_dim
        layers = []
        
        # Calculate input dimensions for CNN
        # For 42-dimensional input, reshape to 6x7 (closest to square)
        self.input_height = 6
        self.input_width = 7
    #    print(f"Input will be reshaped to: {self.input_height}x{self.input_width}", flush=True)
        
        # Handle input layer
        if isinstance(fc_layers[0], tuple):  # CNN layer
    #        print("First layer is CNN", flush=True)
            in_channels, out_channels, kernel_size = fc_layers[0]
            if in_channels != 1:
                raise ValueError(f"First CNN layer must have 1 input channel, got {in_channels}")
    #        print(f"CNN layer params - in_channels: {in_channels}, out_channels: {out_channels}, kernel_size: {kernel_size}", flush=True)
            layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
            layers.append(nn.ReLU(inplace=True))
            # Calculate output size after CNN
            current_dim = out_channels * self.input_height * self.input_width
    #        print(f"After first CNN layer, current_dim: {current_dim}", flush=True)
        else:  # MLP layer
    #        print("First layer is MLP", flush=True)
            layers.append(nn.Linear(inp_dim, fc_layers[0]))
            layers.append(nn.ReLU(inplace=True))
            current_dim = fc_layers[0]
    #        print(f"After first MLP layer, current_dim: {current_dim}", flush=True)z
        # Handle middle layers
        for i in range(1, len(fc_layers)):
    #        print(f"Processing layer {i}: {fc_layers[i]}", flush=True)
            if isinstance(fc_layers[i], tuple):  # CNN layer
                in_channels, out_channels, kernel_size = fc_layers[i]
                # Verify input channels match previous layer's output channels
                if i > 0 and isinstance(fc_layers[i-1], tuple):
                    prev_out_channels = fc_layers[i-1][1]
                    if in_channels != prev_out_channels:
                        raise ValueError(f"CNN layer {i} input channels ({in_channels}) must match previous layer's output channels ({prev_out_channels})")
    #            print(f"CNN layer params - in_channels: {in_channels}, out_channels: {out_channels}, kernel_size: {kernel_size}", flush=True)
                layers.append(nn.Conv2d(in_channels, out_channels, kernel_size, padding=kernel_size//2))
                layers.append(nn.ReLU(inplace=True))
                current_dim = out_channels * self.input_height * self.input_width
    #            print(f"After CNN layer, current_dim: {current_dim}", flush=True)
            else:  # MLP layer
                # If previous layer was CNN, we need to flatten
                if isinstance(fc_layers[i-1], tuple):
    #                print("Previous layer was CNN, adding flatten", flush=True)
                    layers.append(nn.Flatten())
                    # No need to square the dimension since we're already tracking the correct size
    #                print(f"After flatten, current_dim: {current_dim}", flush=True)
                layers.append(nn.Linear(current_dim, fc_layers[i]))
                layers.append(nn.ReLU(inplace=True))
                current_dim = fc_layers[i]
    #                print(f"After MLP layer, current_dim: {current_dim}", flush=True)

        # Handle output layer
        # If last layer was CNN, we need to flatten
        if isinstance(fc_layers[-1], tuple):
    #        print("Last layer was CNN, adding flatten", flush=True)
            layers.append(nn.Flatten())
            # No need to square the dimension since we're already tracking the correct size
    #        print(f"After final flatten, current_dim: {current_dim}", flush=True)
        layers.append(nn.Linear(current_dim, out_dim))
    #    print(f"Final layer: Linear({current_dim}, {out_dim})", flush=True)
        
        self.classifier = nn.Sequential(*layers)
        self.has_conv = any(isinstance(layer, tuple) for layer in fc_layers)
    #    print("MlpNetwork initialization complete", flush=True)

    def forward(self, input):
        if self.has_conv:
            # Reshape input for CNN layers if needed
            batch_size = input.shape[0]
            if len(input.shape) == 2:  # If input is flat
                # Reshape to (batch, channels, height, width)
                input = input.view(batch_size, 1, self.input_height, self.input_width)
        
        out = self.classifier(input)
        return out