import torch
import torch.nn as nn
import torchvision.models as models

from typing import Dict, Optional, Iterable, Hashable, Tuple, Union, List
from collections import OrderedDict

from DatasetWrapper import DatasetWrapper
import utils


class ForwardHookedOutput(nn.Module):
    def __init__(self, base_model: nn.Module, output_layers_specification: Union[Tuple[Hashable], List[Hashable], bool], *args):
        # Init and store base model
        super().__init__(*args)
        self.base_model = base_model

        # Output hooks
        self.output_layers = []
        self.fwd_hooks = []
        self.hook_out = OrderedDict()
        self._module_to_layer_name = {}  # Mapping of modules to layername

        # Register hooks
        for module_name, module in utils.filter_all_named_modules(self.base_model, output_layers_specification, require_leaf=False):
            self._module_to_layer_name[module] = module_name
            self.fwd_hooks.append(
                module.register_forward_hook(self.hook)
            )
            self.output_layers.append(module_name)

    def hook(self, module, inputs, outputs):
        layer_name = self._module_to_layer_name[module]
        assert type(inputs) is tuple and len(inputs) == 1, f"Expected input to be a tuple with length 1, got {inputs}."
        self.hook_out[layer_name] = inputs[0]

    def forward(self, x):
        out = self.base_model(x)
        return out, self.hook_out


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_layers_widths: List[int], output_size: int,
                 use_bias: bool = True, use_softmax: bool = False, use_batch_norm: bool = True):
        super(MLP, self).__init__()

        layers = OrderedDict()

        layers['flatten'] = nn.Flatten()

        if len(hidden_layers_widths) == 0:
            layers['fc'] = nn.Linear(in_features=input_size, out_features=output_size, bias=use_bias)
        else:
            layers['0'] = _MLPBlock(input_dim=input_size, output_dim=hidden_layers_widths[0],
                                    use_bias=use_bias, use_batch_norm=use_batch_norm)
            for idx, (in_size, out_size) in enumerate(zip(hidden_layers_widths[:-1], hidden_layers_widths[1:])):
                layers[f'{idx+1}'] = _MLPBlock(input_dim=in_size, output_dim=out_size,
                                               use_bias=use_bias, use_batch_norm=use_batch_norm)
            layers['fc'] = nn.Linear(in_features=hidden_layers_widths[-1], out_features=output_size, bias=use_bias)

        if use_softmax:
            layers['softmax'] = nn.Softmax()

        self.model = nn.Sequential(
            layers
        )

    def forward(self, x):
        out = self.model(x)
        return out


class _MLPBlock(nn.Module):
    def __init__(self, input_dim: int, output_dim: int, use_bias: bool = None, use_batch_norm: bool = False):
        super(_MLPBlock, self).__init__()
        self.fc = nn.Linear(in_features=input_dim, out_features=output_dim, bias=use_bias)
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn = nn.BatchNorm1d(num_features=output_dim)
        self.relu = nn.ReLU()

    def forward(self, x):
        x = self.fc(x)
        if self.use_batch_norm:
            x = self.bn(x)
        x = self.relu(x)
        return x


def get_model(model_cfg: Dict, datasetwrapper: DatasetWrapper):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name: str = model_cfg['model-name'].lower()

    if model_name == 'resnet18':
        base_model = models.resnet18(pretrained=False)
        # Set input channels to match input channels of dataset
        data_input_channels = datasetwrapper.input_batch_shape[1]
        if base_model.conv1.in_channels != data_input_channels:
            # base_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            old_layer = base_model.conv1
            base_model.conv1 = nn.Conv2d(
                in_channels=data_input_channels,
                out_channels=old_layer.out_channels, kernel_size=old_layer.kernel_size,
                stride=old_layer.stride, padding=old_layer.padding, bias=old_layer.bias
            )

        base_model.fc = nn.Linear(in_features=base_model.fc.in_features, out_features=datasetwrapper.num_classes)
    elif model_name.startswith('mlp'):
        # hidden_layer_sizes = [128, 128, 64, 64]
        hidden_layer_sizes = [512] * 5 if '_large' not in model_name else [1024] * 10
        base_model = MLP(
            input_size=datasetwrapper.input_batch_shape[1:].numel(),
            hidden_layers_widths=hidden_layer_sizes,
            output_size=datasetwrapper.num_classes,
            use_batch_norm='_nobn' not in model_name
        )
    else:
        assert model_cfg['model-name'].lower() in ('resnet18', 'mlp', 'mlp_bn', 'mlp_large', 'mlp_large_bn'),\
            f"Model type not supported: {model_cfg['model-name']}"
        raise NotImplementedError

    base_model.to(device)
    out_layers = model_cfg['embedding_layers']  # TODO(marius): Make support "True" for all layers
    print(base_model)
    ret_model = ForwardHookedOutput(base_model, out_layers).to(device)
    print("Tracking layers: ", end='\n\t')
    print(*ret_model.output_layers, sep=',\n\t')
    return ret_model


def get_resnet18_model(model_cfg: Dict):
    # TODO(marius): Add support for mnist single channel input data
    # base_model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    # base_model.to(device)

    # out_layers = {f'layer{i}': 1 for i in range(3, 5)}
    # out_layers = model_cfg['embedding_layers']
    # ret_model = ForwardHookedOutput(base_model, out_layers).to(device)
    # return ret_model
    raise NotImplementedError()


if __name__ == "__main__":
    _ret_model = get_resnet18_model(dict(
        embedding_layers={'layer3': 1, 'layer4': 1}
    ))
    print(_ret_model)
    pass

