from collections import OrderedDict

import torch
from torch import nn
from torch.nn import functional as F
from torchsummary import summary
from fvcore.common.registry import Registry

from arnet import utils

MODEL_REGISTRY = Registry("MODEL")
MODEL_REGISTRY.__doc__ = """Registry for video modeling."""


SETTINGS = {
    'c3d': {
        'out_channels': [8, 8, 16], # more channels to compensate for 5 -> 121
        'kernels': [[5, 5, 5], [3, 3, 3], [3, 3, 3]],
        'paddings': [[2, 2, 2], [1, 1, 1], [1, 1, 1]],
        'poolings': [[2, 4, 4], [2, 2, 2], [2, 2, 2]],
        'out_features': [512, 64, 32],
    },
    'c2d': {
        'out_channels': [4, 8, 16],
        'kernels': [[121, 5, 5], [1, 3, 3], [1, 3, 3]],
        'paddings': [[0, 2, 2], [0, 1, 1], [0, 1, 1]],
        'poolings': [[1, 4, 4], [1, 2, 2], [1, 2, 2]],
        'out_features': [512, 64, 32],
    },
    'cnn': {
        'out_channels': [16, 16, 16],
        'kernels': [[1, 5, 5], [1, 3, 3], [1, 3, 3]],
        'paddings': [[0, 2, 2], [0, 1, 1], [0, 1, 1]],
        'poolings': [[1, 2, 2], [1, 2, 2], [1, 2, 2]],
        'out_features': [64, 32],
    },
    'fusion': {
        'out_channels': [8, 8, 16],  # more channels to compensate for 5 -> 121
        'kernels': [[5, 5, 5], [3, 3, 3], [3, 3, 3]],
        'paddings': [[2, 2, 2], [1, 1, 1], [1, 1, 1]],
        'poolings': [[2, 4, 4], [2, 2, 2], [2, 2, 2]],
        'out_features': [512, 64],
    },
    'fusion_cnn': {
        'out_channels': [16, 16, 16],
        'kernels': [[1, 5, 5], [1, 3, 3], [1, 3, 3]],
        'paddings': [[0, 2, 2], [0, 1, 1], [0, 1, 1]],
        'poolings': [[1, 2, 2], [1, 2, 2], [1, 2, 2]],
        'out_features': [64, 32, 4],
        'fuse_point': 1,
    },
}


@MODEL_REGISTRY.register()
class SimpleC3D(nn.Module):
    mode = 'classification'

    def __init__(self, cfg):
        """
        Args:
            input_shape (tuple): (C,T,H,W)
        """
        super().__init__()
        self.set_class_weight(cfg)
        self.result = {}

        input_shape = (1, cfg.DATA.NUM_FRAMES, cfg.DATA.HEIGHT, cfg.DATA.WIDTH)  # (1, 121, 64, 128)
        kernels = SETTINGS[cfg.LEARNER.MODEL.SETTINGS]['kernels']
        paddings = SETTINGS[cfg.LEARNER.MODEL.SETTINGS]['paddings']
        poolings = SETTINGS[cfg.LEARNER.MODEL.SETTINGS]['poolings']
        out_channels = SETTINGS[cfg.LEARNER.MODEL.SETTINGS]['out_channels']
        out_features = SETTINGS[cfg.LEARNER.MODEL.SETTINGS]['out_features']

        self.input_shape = input_shape  # needed when testing
        self.convs = nn.Sequential(OrderedDict([
            ('conv1', nn.Conv3d(input_shape[0], out_channels[0], kernels[0], padding=paddings[0])),
            ('relu1', nn.LeakyReLU()),
            ('pool1', nn.MaxPool3d(poolings[0])),
            #('bn1',   nn.BatchNorm3d(out_channels[0])),

            ('conv2', nn.Conv3d(out_channels[0], out_channels[1], kernels[1], padding=paddings[1])),
            ('relu2', nn.LeakyReLU()),
            ('pool2', nn.MaxPool3d(poolings[1])),
            #('bn2',   nn.BatchNorm3d(out_channels[1])),

            ('conv3', nn.Conv3d(out_channels[1], out_channels[2], kernels[2], padding=paddings[2])),
            ('relu3', nn.LeakyReLU()),
            ('pool3', nn.MaxPool3d(poolings[2])),
            #('bn3',   nn.BatchNorm3d(out_channels[2])),
        ]))
        fc_input_dim = self.infer_output_shape(self.convs, input_shape).numel()

        out_features = [fc_input_dim] + out_features
        self.linears = nn.Sequential()
        for i in range(1, len(out_features)):
            self.linears.add_module(f'linear{i}', nn.Linear(out_features[i-1], out_features[i]))
            self.linears.add_module(f'relu_lin{i}', nn.ReLU())
            #self.linears.add_module(f'bn_lin{i}', nn.BatchNorm1d(out_features[i]))
        self.linears.add_module(f'linear{len(out_features)}', nn.Linear(out_features[-1], 2))

        print(self.infer_output_shape(self.convs, input_shape), fc_input_dim)
        summary(self.cuda(), (input_shape))

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

    def get_loss(self, output, target):
        log_prob = F.log_softmax(output, dim=-1)
        loss = F.nll_loss(log_prob, target, weight=self.class_weight, reduction='mean')

        self.result['y_true'] = target
        self.result['y_prob'] = torch.exp(log_prob[:,1])

        return loss


@MODEL_REGISTRY.register()
class SimpleLSTM(nn.Module):
    mode = 'classification'

    def __init__(self, cfg):
        super().__init__()
        self.set_class_weight(cfg)
        self.result = {}

        self.lstm = nn.LSTM(
            input_size=len(cfg.DATA.FEATURES),
            hidden_size=50,
            num_layers=2,
            batch_first=True,
            #dropout=0.5,
        )
        self.linear = nn.Linear(50, 2)
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

    def get_loss(self, output, target):
        log_prob = F.log_softmax(output, dim=-1)
        loss = F.nll_loss(log_prob, target, weight=self.class_weight, reduction='mean')

        self.result['y_true'] = target
        self.result['y_prob'] = torch.exp(log_prob[:,1])

        return loss


@MODEL_REGISTRY.register()
class FusionNet(nn.Module):
    mode = 'classification'

    def __init__(self, cfg):
        """
        Args:
            input_shape (tuple): (C,T,H,W)
        """
        super().__init__()
        self.set_class_weight(cfg)
        self.result = {}

        input_shape = (1, cfg.DATA.NUM_FRAMES, cfg.DATA.HEIGHT, cfg.DATA.WIDTH)
        s = SETTINGS[cfg.LEARNER.MODEL.SETTINGS]
        s['out_features'].append(2)

        # Convolution layers
        convs = OrderedDict()
        out_prev = input_shape[0]
        for i, (out, kern, pad, pool) in enumerate(zip(s['out_channels'], s['kernels'], s['paddings'], s['poolings'])):
            convs[f'conv{i+1}'] = nn.Conv3d(out_prev, out, kern, padding=pad)
            convs[f'conv_relu{i+1}'] = nn.LeakyReLU()
            convs[f'conv_pool{i+1}'] = nn.MaxPool3d(pool)
            convs[f'conv_bn{i+1}'] = nn.BatchNorm3d(out)
            out_prev = out
        self.convs = nn.Sequential(convs)

        # Linear layers
        linears = OrderedDict()
        out_prev = self.infer_output_shape(self.convs, input_shape).numel()
        for i, out in enumerate(s['out_features'][:s['fuse_point']+1]):
            linears[f'linear{i+1}'] = nn.Linear(out_prev, out)
            linears[f'linear_relu{i+1}'] = nn.LeakyReLU()
            linears[f'linear_bn{i+1}'] = nn.BatchNorm1d(out)
            out_prev = out
        self.linears = nn.Sequential(linears)

        # Fused layers
        fused = OrderedDict()
        out_prev += 2
        for i, out in enumerate(s['out_features'][s['fuse_point']+1:]):
            fused[f'fused{i+1}'] = nn.Linear(out_prev, out)
            fused[f'fused_relu{i+1}'] = nn.LeakyReLU()
            fused[f'fused_bn{i+1}'] = nn.BatchNorm1d(out)
            out_prev = out
        self.fused = nn.Sequential(fused)

        summary(self.cuda(), [input_shape, (2,)])

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
        image, size, target, meta = batch
        x = self(image, size)
        log_prob = F.log_softmax(x, dim=-1)
        loss = F.nll_loss(log_prob, target, weight=self.class_weight, reduction='mean')

        self.result['image'] = image
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
