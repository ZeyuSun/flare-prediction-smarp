#!/usr/bin/env python3
# Copyright (c) Facebook, Inc. and its affiliates. All Rights Reserved.
# Adapted from https://github.com/facebookresearch/SlowFast/blob/master/slowfast/visualization/gradcam_utils.py
from operator import attrgetter

import matplotlib.pyplot as plt
import torch
import torch.nn.functional as F


def revert_tensor_normalize(tensor, mean, std):
    """
    Revert normalization for a given tensor by multiplying by the std and adding the mean.
    Args:
        tensor (tensor): tensor to revert normalization.
        mean (tensor or list): mean value to add.
        std (tensor or list): std to multiply.
    """
    if type(mean) == list:
        mean = torch.tensor(mean)
    if type(std) == list:
        std = torch.tensor(std)
    tensor = tensor * std
    tensor = tensor + mean
    return tensor


def get_layer(model, layer_name):
    """
    Return the targeted layer (nn.Module Object) given a hierarchical layer name,
    separated by /.
    Args:
        model (model): model to get layers from.
        layer_name (str): name of the layer.
    Returns:
        prev_module (nn.Module): the layer from the model with `layer_name` name.
    """

    layer = attrgetter(layer_name)(model)

    return layer


class GradCAM:
    """
    GradCAM class helps create localization maps using the Grad-CAM method for input videos
    and overlap the maps over the input videos as heatmaps.
    https://arxiv.org/pdf/1610.02391.pdf
    """

    def __init__(
        self, model, target_layers, data_mean, data_std, colormap="Reds" #"viridis"
    ):
        """
        Args:
            model (model): the model to be used.
            target_layers (list of str(s)): name of convolutional layer to be used to get
                gradients and feature maps from for creating localization maps.
            data_mean (tensor or list): mean value to add to input videos.
            data_std (tensor or list): std to multiply for input videos.
            colormap (Optional[str]): matplotlib colormap used to create heatmap.
                See https://matplotlib.org/3.3.0/tutorials/colors/colormaps.html
        """

        self.model = model
        # Run in eval mode.
        self.model.eval()
        self.target_layers = target_layers

        self.gradients = {}
        self.activations = {}
        self.colormap = plt.get_cmap(colormap)
        self.data_mean = data_mean
        self.data_std = data_std
        self._register_hooks()

    def _register_single_hook(self, layer_name):
        """
        Register forward and backward hook to a layer, given layer_name,
        to obtain gradients and activations.
        Args:
            layer_name (str): name of the layer.
        """

        def get_gradients(module, grad_input, grad_output):
            self.gradients[layer_name] = grad_output[0].detach()

        def get_activations(module, input, output):
            self.activations[layer_name] = output.clone().detach()

        target_layer = get_layer(self.model, layer_name=layer_name)
        target_layer.register_forward_hook(get_activations)
        target_layer.register_backward_hook(get_gradients)

    def _register_hooks(self):
        """
        Register hooks to layers in `self.target_layers`.
        """
        for layer_name in self.target_layers:
            self._register_single_hook(layer_name=layer_name)

    def _calculate_localization_map(self, inputs, labels=None):
        """
        Calculate localization map for all inputs with Grad-CAM.
        Args:
            inputs (list of tensor(s)): the input clips.
            labels (Optional[tensor]): labels of the current input clips.
        Returns:
            localization_maps (list of ndarray(s)): the localization map for
                each corresponding input.
            preds (tensor): shape (n_instances, n_class). Model predictions for `inputs`.
        """
        with torch.enable_grad():
            preds = self.model(inputs.clone())

            if labels is None:
                score = torch.max(preds, dim=-1)[0]  # This works when thresh = 0.5
            else:
                if labels.ndim == 1:
                    labels = labels.unsqueeze(-1)
                score = torch.gather(preds, dim=1, index=labels)

            self.model.zero_grad()
            score = torch.sum(score)
            score.backward()

        _, _, T, H, W = inputs.size()

        gradients = self.gradients[self.target_layers[0]]
        activations = self.activations[self.target_layers[0]]
        B, C, Tg, _, _ = gradients.size()

        weights = torch.mean(gradients.view(B, C, Tg, -1), dim=3)

        weights = weights.view(B, C, Tg, 1, 1)
        localization_map = torch.sum(
            weights * activations, dim=1, keepdim=True
        )
        localization_map = F.relu(localization_map)
        localization_map = F.interpolate(
            localization_map,
            size=(T, H, W),
            mode="trilinear",
            align_corners=False,
        )
        localization_map_min, localization_map_max = (
            torch.min(localization_map.view(B, -1), dim=-1, keepdim=True)[
                0
            ],
            torch.max(localization_map.view(B, -1), dim=-1, keepdim=True)[
                0
            ],
        )
        localization_map_min = torch.reshape(
            localization_map_min, shape=(B, 1, 1, 1, 1)
        )
        localization_map_max = torch.reshape(
            localization_map_max, shape=(B, 1, 1, 1, 1)
        )
        # Normalize the localization map.
        localization_map = (localization_map - localization_map_min) / (
            localization_map_max - localization_map_min + 1e-6
        )
        localization_map = localization_map.data

        return localization_map, preds

    def __call__(self, inputs, labels=None, alpha=0.5):
        """
        Visualize the localization maps on their corresponding inputs as heatmap,
        using Grad-CAM.
        Args:
            inputs (list of tensor(s)): the input clips.
            labels (Optional[tensor]): labels of the current input clips.
            alpha (float): transparency level of the heatmap, in the range [0, 1].
        Returns:
            result_ls (list of tensor(s)): the visualized inputs.
            preds (tensor): shape (n_instances, n_class). Model predictions for `inputs`.
        """
        #import matplotlib; matplotlib.use('Qt5Agg')
        localization_map, preds = self._calculate_localization_map(
            inputs, labels=labels
        )
        # Convert (B, 1, T, H, W) to (B, T, H, W)
        localization_map = localization_map.squeeze(dim=1)
        if localization_map.device != torch.device("cpu"):
            localization_map = localization_map.cpu()
        heatmap = self.colormap(localization_map.numpy())
        heatmap = heatmap[:, :, :, :, :3]
        # Permute input from (B, C, T, H, W) to (B, T, H, W, C)
        x = inputs.permute(0, 2, 3, 4, 1)
        if x.device != torch.device("cpu"):
            x = x.cpu()
        x = revert_tensor_normalize(
            x, self.data_mean, self.data_std
        )
        from arnet import utils
        mean, std = x.mean(), x.std()
        x = utils.array_to_float_video(x, low=mean-3*std, high=mean+3*std, perc=False)
        heatmap = torch.from_numpy(heatmap)
        x_gradcam = alpha * heatmap + (1 - alpha) * x # channel of x expands from 1 to 3
        # Permute inp back to (B, C, T, H, W)
        x_gradcam = x_gradcam.permute(0, 4, 1, 2, 3)
        x = x.permute(0, 4, 1, 2, 3)
        heatmap = heatmap.permute(0, 4, 1, 2, 3)
        # All returned maps have range [0, 1]
        return x_gradcam, x, heatmap, preds
