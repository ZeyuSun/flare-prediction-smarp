from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchinfo import summary
from fvcore.common.registry import Registry

from arnet import utils

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """Registry for video modeling."""


SETTINGS = {
    'c3d': {
        'out_channels': [32, 32, 64, 64, 64], # more channels to compensate for 5 -> 121
        'kernels': [[5, 11, 11], [3, 5, 5], [3, 3, 3], [3, 3, 3], [3, 3, 3]],
        'paddings': [[2, 5, 5], [1, 2, 2], [1, 1, 1], [1, 1, 1], [1, 1, 1]],
        'poolings': [[2, 4, 4], [2, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2]],
        'out_features': [128, 64, 16],
    },
    'c2d': {
        'out_channels': [4, 8, 16],
        'kernels': [[121, 5, 5], [1, 3, 3], [1, 3, 3]],
        'paddings': [[0, 2, 2], [0, 1, 1], [0, 1, 1]],
        'poolings': [[1, 4, 4], [1, 2, 2], [1, 2, 2]],
        'out_features': [512, 64, 32],
    },
    'cnn': {
        'out_channels': [64, 64, 64, 64, 64],
        'kernels': [[1, 11, 11], [1, 5, 5], [1, 3, 3], [1, 3, 3], [1, 3, 3]],
        'paddings': [[0, 5, 5], [0, 2, 2], [0, 1, 1], [0, 1, 1], [0, 1, 1]],
        'poolings': [[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2]],
        'out_features': [128, 64, 16],
    },
    'cnn_li2020': {
        'out_channels': [64, 64, 64, 64, 64],
        'kernels': [[1, 11, 11], [1, 11, 11], [1, 3, 3], [1, 3, 3], [1, 3, 3]],
        'paddings': [[0, 5, 5], [0, 5, 5], [0, 1, 1], [0, 1, 1], [0, 1, 1]],
        'poolings': [[1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2], [1, 2, 2]],
        'out_features': [128, 64],
    },
    'fusion_c3d': {
        'out_channels': [8, 8, 16],  # more channels to compensate for 5 -> 121
        'kernels': [[5, 5, 5], [3, 3, 3], [3, 3, 3]],
        'paddings': [[2, 2, 2], [1, 1, 1], [1, 1, 1]],
        'poolings': [[2, 4, 4], [2, 2, 2], [2, 2, 2]],
        'out_features': [512, 64, 8],
        'fuse_point': 2,
    },
    'fusion_cnn': {
        'out_channels': [16, 16, 16],
        'kernels': [[1, 5, 5], [1, 3, 3], [1, 3, 3]],
        'paddings': [[0, 2, 2], [0, 1, 1], [0, 1, 1]],
        'poolings': [[1, 2, 2], [1, 2, 2], [1, 2, 2]],
        'out_features': [64, 32, 8],
        'fuse_point': 2,
    },
}


@MODEL_REGISTRY.register()
class SimpleC3D(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            input_shape (tuple): (C,T,H,W)
        """
        super().__init__()
        self.set_class_weight(cfg)
        self.result = {}

        input_shape = (1, cfg.DATA.NUM_FRAMES, cfg.DATA.HEIGHT, cfg.DATA.WIDTH)
        s = SETTINGS[cfg.LEARNER.MODEL.SETTINGS].copy()

        # Convolution layers
        convs = OrderedDict()
        out_prev = input_shape[0]
        for i, (out, kern, pad, pool) in enumerate(zip(s['out_channels'], s['kernels'], s['paddings'], s['poolings'])):
            convs[f'conv{i+1}'] = nn.Conv3d(out_prev, out, kern, padding=pad)
            convs[f'conv_relu{i+1}'] = nn.LeakyReLU()
            convs[f'conv_pool{i+1}'] = nn.MaxPool3d(pool)
            #convs[f'conv_bn{i+1}'] = nn.BatchNorm3d(out)
            out_prev = out
        self.convs = nn.Sequential(convs)

        # Linear layers
        linears = OrderedDict()
        out_prev = self.infer_output_shape(self.convs, input_shape).numel()
        out_dims = s['out_features'] + [2]
        for i, out in enumerate(out_dims):
            linears[f'linear{i + 1}'] = nn.Linear(out_prev, out)
            if i == len(out_dims) - 1:
                break
            linears[f'linear_relu{i + 1}'] = nn.LeakyReLU()
            #linears[f'linear_bn{i + 1}'] = nn.BatchNorm1d(out)
            out_prev = out
        self.linears = nn.Sequential(linears)

        #summary(self, input_shape)

    def set_class_weight(self, cfg):
        if cfg.LEARNER.CLASS_WEIGHT is None:
            class_weight = [1, 1]
        elif cfg.LEARNER.CLASS_WEIGHT == 'balanced':
            class_weight = [1 / w for w in cfg.DATA.CLASS_WEIGHT]
        else:
            class_weight = cfg.LEARNER.CLASS_WEIGHT
        self.register_buffer('class_weight', torch.tensor(class_weight, dtype=torch.float))

    def infer_output_shape(self, model, input_shape):
        input = torch.zeros(1, *input_shape)
        output = model(input)
        return output.shape[1:]

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.linears(x)
        return x

    def get_loss(self, batch):
        video, size, target, meta = batch
        output = self(video)

        log_prob = F.log_softmax(output, dim=-1)
        loss = F.nll_loss(log_prob, target, weight=self.class_weight, reduction='mean')

        self.result['video'] = video
        self.result['meta'] = meta
        self.result['y_true'] = target
        self.result['y_prob'] = torch.exp(log_prob[:,1])

        return loss


@MODEL_REGISTRY.register()
class CNN_Li2020(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            input_shape (tuple): (C,T,H,W)
        """
        super().__init__()
        self.set_class_weight(cfg)
        self.result = {}

        input_shape = (1, cfg.DATA.NUM_FRAMES, cfg.DATA.HEIGHT, cfg.DATA.WIDTH)
        s = SETTINGS[cfg.LEARNER.MODEL.SETTINGS].copy()

        # Convolution layers
        convs = OrderedDict()
        out_prev = input_shape[0]
        for i, (out, kern, pad, pool) in enumerate(zip(s['out_channels'], s['kernels'], s['paddings'], s['poolings'])):
            convs[f'conv{i + 1}'] = nn.Conv3d(out_prev, out, kern, padding=pad)
            convs[f'conv_bn{i+1}'] = nn.BatchNorm3d(out)
            convs[f'conv_relu{i + 1}'] = nn.ReLU()
            convs[f'conv_pool{i + 1}'] = nn.MaxPool3d(pool)
            out_prev = out
        self.convs = nn.Sequential(convs)

        # Linear layers
        linears = OrderedDict()
        out_prev = self.infer_output_shape(self.convs, input_shape).numel()
        out_dims = s['out_features'] + [2]
        for i, out in enumerate(out_dims):
            linears[f'linear{i + 1}'] = nn.Linear(out_prev, out)
            if i == len(out_dims) - 1:
                break
            #linears[f'linear_relu{i + 1}'] = nn.ReLU()
            linears[f'linear_bn{i + 1}'] = nn.BatchNorm1d(out)
            linears[f'linear_dropout{i + 1}'] = nn.Dropout(p=0.5)
            out_prev = out
        self.linears = nn.Sequential(linears)

        #summary(self, input_shape)

    def set_class_weight(self, cfg):
        if cfg.LEARNER.CLASS_WEIGHT is None:
            class_weight = [1, 1]
        elif cfg.LEARNER.CLASS_WEIGHT == 'balanced':
            class_weight = [1 / w for w in cfg.DATA.CLASS_WEIGHT]
        else:
            class_weight = cfg.LEARNER.CLASS_WEIGHT
        self.register_buffer('class_weight', torch.tensor(class_weight, dtype=torch.float))

    def infer_output_shape(self, model, input_shape):
        input = torch.zeros(1, *input_shape) # device!
        output = model(input)
        return output.shape[1:]

    def forward(self, x):
        x = self.convs(x)
        x = torch.flatten(x, 1)
        x = self.linears(x)
        return x

    def get_loss(self, batch):
        video, size, target, meta = batch
        output = self(video)

        log_prob = F.log_softmax(output, dim=-1)
        loss = F.nll_loss(log_prob, target, weight=self.class_weight, reduction='mean')

        self.result['video'] = video
        self.result['meta'] = meta
        self.result['y_true'] = target
        self.result['y_prob'] = torch.exp(log_prob[:, 1])

        return loss


@MODEL_REGISTRY.register()
class MLP(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.set_class_weight(cfg)
        self.result = {}

        linears = OrderedDict()
        out_prev = len(cfg.DATA.FEATURES)
        out_dims = [64, 32, 32, 8, 2]
        for i, out in enumerate(out_dims):
            linears[f'linear{i+1}'] = nn.Linear(out_prev, out)
            if i == len(out_dims) - 1:
                break
            linears[f'linear_relu{i+1}'] = nn.LeakyReLU()
            #linears[f'linear_bn{i+1}'] = nn.BatchNorm1d(out)
            out_prev = out
        self.linears = nn.Sequential(linears)

    def set_class_weight(self, cfg):
        if cfg.LEARNER.CLASS_WEIGHT is None:
            class_weight = [1, 1]
        elif cfg.LEARNER.CLASS_WEIGHT == 'balanced':
            class_weight = [1 / w for w in cfg.DATA.CLASS_WEIGHT]
        else:
            class_weight = cfg.LEARNER.CLASS_WEIGHT
        self.register_buffer('class_weight', torch.tensor(class_weight, dtype=torch.float))

    def forward(self, x):
        x = self.linears(x)
        return x

    def get_loss(self, batch):
        x, target, meta = batch
        x = x[:,-1]  # x has shape [Batchsize, Depth, Dims] with Depth=1
        output = self(x)

        log_prob = F.log_softmax(output, dim=-1)
        loss = F.nll_loss(log_prob, target, weight=self.class_weight, reduction='mean')

        self.result['x'] = x
        self.result['meta'] = meta
        self.result['y_true'] = target
        self.result['y_prob'] = torch.exp(log_prob[:, 1])

        return loss


@MODEL_REGISTRY.register()
class SimpleLSTM(nn.Module):
    def __init__(self, cfg):
        super().__init__()
        self.set_class_weight(cfg)
        self.result = {}

        self.lstm = nn.LSTM(
            input_size=len(cfg.DATA.FEATURES),
            hidden_size=64,
            num_layers=2,
            batch_first=True,
            #dropout=0.5,
        )
        self.linear = nn.Linear(64, 2)
        #self.loss = nn.BCELoss(weight=) # weight has to be of size batch_num
        # CrossEntropyLoss = LogSoftmax + NLLLoss
        # self.loss_func = nn.CrossEntropyLoss(
        #     weight=self.class_weight,
        # )

    def set_class_weight(self, cfg):
        if cfg.LEARNER.CLASS_WEIGHT is None:
            class_weight = [1, 1]
        elif cfg.LEARNER.CLASS_WEIGHT == 'balanced':
            class_weight = [1 / w for w in cfg.DATA.CLASS_WEIGHT]
        else:
            class_weight = cfg.LEARNER.CLASS_WEIGHT
        self.register_buffer('class_weight', torch.tensor(class_weight, dtype=torch.float))

    def forward(self, x):
        x, (hn, cn) = self.lstm(x, None) # x.shape = [N (batch), T (seq), C (channels)]
        x = self.linear(x[:,-1,:])
        return x

    def get_loss(self, batch):
        x, target, meta = batch
        output = self(x)

        log_prob = F.log_softmax(output, dim=-1)
        loss = F.nll_loss(log_prob, target, weight=self.class_weight, reduction='mean')

        self.result['x'] = x
        self.result['meta'] = meta
        self.result['y_true'] = target
        self.result['y_prob'] = torch.exp(log_prob[:,1])

        return loss


@MODEL_REGISTRY.register()
class FusionNet(nn.Module):
    def __init__(self, cfg):
        """
        Args:
            input_shape (tuple): (C,T,H,W)
        """
        super().__init__()
        self.set_class_weight(cfg)
        self.result = {}

        input_shape = (1, cfg.DATA.NUM_FRAMES, cfg.DATA.HEIGHT, cfg.DATA.WIDTH)
        s = SETTINGS[cfg.LEARNER.MODEL.SETTINGS].copy()

        # Convolution layers
        convs = OrderedDict()
        out_prev = input_shape[0]
        for i, (out, kern, pad, pool) in enumerate(zip(s['out_channels'], s['kernels'], s['paddings'], s['poolings'])):
            convs[f'conv{i+1}'] = nn.Conv3d(out_prev, out, kern, padding=pad)
            convs[f'conv_relu{i+1}'] = nn.LeakyReLU()
            convs[f'conv_pool{i+1}'] = nn.MaxPool3d(pool)
            #convs[f'conv_bn{i+1}'] = nn.BatchNorm3d(out)
            out_prev = out
        self.convs = nn.Sequential(convs)

        # Linear layers
        linears = OrderedDict()
        out_prev = self.infer_output_shape(self.convs, input_shape).numel()
        for i, out in enumerate(s['out_features'][:s['fuse_point']+1]):
            linears[f'linear{i+1}'] = nn.Linear(out_prev, out)
            linears[f'linear_relu{i+1}'] = nn.LeakyReLU()
            #linears[f'linear_bn{i+1}'] = nn.BatchNorm1d(out)
            out_prev = out
        self.linears = nn.Sequential(linears)

        # Fused layers
        fused = OrderedDict()
        out_prev += 2
        out_dims = s['out_features'][s['fuse_point'] + 1:] + [2]
        for i, out in enumerate(out_dims):
            fused[f'fused{i+1}'] = nn.Linear(out_prev, out)
            if i == len(out_dims) - 1:
                break
            fused[f'fused_relu{i+1}'] = nn.LeakyReLU()
            #fused[f'fused_bn{i+1}'] = nn.BatchNorm1d(out)
            out_prev = out
        self.fused = nn.Sequential(fused)

        #summary(self, [input_shape, (2,)])

    def set_class_weight(self, cfg):
        if cfg.LEARNER.CLASS_WEIGHT is None:
            class_weight = [1, 1]
        elif cfg.LEARNER.CLASS_WEIGHT == 'balanced':
            class_weight = [1 / w for w in cfg.DATA.CLASS_WEIGHT]
        else:
            class_weight = cfg.LEARNER.CLASS_WEIGHT
        self.register_buffer('class_weight', torch.tensor(class_weight, dtype=torch.float))

    def infer_output_shape(self, model, input_shape):
        input = torch.zeros(1, *input_shape)
        output = model(input)
        return output.shape[1:]

    def forward(self, image, size):
        x1 = self.convs(image)
        x1 = torch.flatten(x1, 1)
        x1 = self.linears(x1)

        x = torch.cat((x1, size), dim=1)
        x = self.fused(x)
        return x

    def get_loss(self, batch):
        video, size, target, meta = batch
        x = self(video, size)
        log_prob = F.log_softmax(x, dim=-1)
        loss = F.nll_loss(log_prob, target, weight=self.class_weight, reduction='mean')

        self.result['video'] = video
        self.result['meta'] = meta
        self.result['y_true'] = target
        self.result['y_prob'] = torch.exp(log_prob[:, 1])

        return loss


def build_model(cfg):
    name = cfg.LEARNER.MODEL.NAME
    model = MODEL_REGISTRY.get(name)(cfg)
    return model

def get_model_mode(model):
    if isinstance(model, str):
        return MODEL_REGISTRY.get(model).mode
    elif isinstance(model, nn.Module):
        return model.mode
