import torch
import torch.nn as nn
import torchvision.models

from typing import Dict, Hashable, Tuple, Union, List
from collections import OrderedDict

from DatasetWrapper import DatasetWrapper
import utils


class ForwardHookedOutput(nn.Module):
    def __init__(self, base_model: nn.Module, output_layers_specification: Union[Tuple[Hashable], List[Hashable], bool]):
        # Init and store base model
        super().__init__()
        self.base_model = base_model

        # Output hooks
        self.output_layers = []
        self.fwd_hooks = []
        self.hook_out = OrderedDict()
        self._module_to_layer_name = {}  # Mapping of modules to layername

        # Register hooks
        for module_name, module in utils.filter_all_named_modules(self.base_model, output_layers_specification):
            self._module_to_layer_name[module] = module_name
            self.fwd_hooks.append(
                module.register_forward_hook(self.hook)
            )
            self.output_layers.append(module_name)

    def hook(self, module, inputs, outputs):
        layer_name = self._module_to_layer_name[module]
        assert type(inputs) is tuple and len(inputs) == 1,\
            f"Expected input to be a tuple with length 1, got {inputs}."
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
            layers['block0'] = _MLPBlock(input_dim=input_size, output_dim=hidden_layers_widths[0],
                                         use_bias=use_bias, use_batch_norm=use_batch_norm)
            for idx, (in_size, out_size) in enumerate(zip(hidden_layers_widths[:-1], hidden_layers_widths[1:])):
                layers[f'block{idx+1}'] = _MLPBlock(input_dim=in_size, output_dim=out_size,
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
    model_name: str = model_cfg['model-name'].lower()

    if model_name == 'resnet18':
        base_model = torchvision.models.resnet18(pretrained=False)
        # Set input channels to match input channels of dataset
        data_input_channels = datasetwrapper.input_batch_shape[1]
        old_layer = base_model.conv1
        if old_layer.in_channels != data_input_channels:
            # base_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            base_model.conv1 = nn.Conv2d(
                in_channels=data_input_channels,
                out_channels=old_layer.out_channels, kernel_size=old_layer.kernel_size,
                stride=old_layer.stride, padding=old_layer.padding, bias=old_layer.bias is not None
            )
        # Set output to match class number of dataset
        base_model.fc = nn.Linear(in_features=base_model.fc.in_features, out_features=datasetwrapper.num_classes)

    elif model_name.startswith('mlp'):
        # Find specified MLP hidden layers
        suffix = model_name[3:]
        sizes = {
            '_linear': [],
            '_single': [512]*1,
            '_small': [256]*4,
            '_wide': [1024]*4,
            '_xwide': [2048]*4,
            '_default': [512]*5,
            '_large': [1024]*10,
        }
        sizes_match = ''
        for sizes_key in sizes.keys():
            if sizes_key in suffix:
                assert not sizes_match, f"Multiple matches for given MLP suffix: {sizes_key} and {sizes_match}."
                sizes_match = sizes_key
        if not sizes_match:
            sizes_match = '_default'
        hidden_layer_sizes = sizes[sizes_match]

        # Construct MLP
        base_model = MLP(
            input_size=datasetwrapper.input_batch_shape[1:].numel(),
            hidden_layers_widths=hidden_layer_sizes,
            output_size=datasetwrapper.num_classes,
            use_bias='_bias' in model_name,
            use_batch_norm='_nobn' not in model_name
        )

    elif model_name.startswith('vgg'):
        # Find specified vgg in torchvision.models
        assert hasattr(torchvision.models, model_name), f"Model type not supported: {model_name}" \
                                                        f"\nNo such VGG model in torchvision.models"
        base_model = getattr(torchvision.models, model_name)(pretrained=False)

        # Check input channels and change layer if needed
        data_input_channels = datasetwrapper.input_batch_shape[1]
        old_layer = base_model.features[0]
        if old_layer.in_channels != data_input_channels:
            # base_model.conv1 = nn.Conv2d(3, 64, kernel_size=(7, 7), stride=(2, 2), padding=(3, 3), bias=False)
            base_model.features[0] = nn.Conv2d(
                in_channels=data_input_channels,
                out_channels=old_layer.out_channels, kernel_size=old_layer.kernel_size,
                stride=old_layer.stride, padding=old_layer.padding, bias=old_layer.bias is not None
            )
        # Set output number of classes
        base_model.classifier[-1] = nn.Linear(in_features=base_model.classifier[-1].in_features,
                                              out_features=datasetwrapper.num_classes)
    else:
        raise NotImplementedError(f"Model type not supported: {model_name}")

    # Set hooked output layers
    out_layers = model_cfg['embedding_layers']
    print(base_model)
    ret_model = ForwardHookedOutput(base_model, out_layers)
    print("Tracking layers: ", end='\n\t')
    print(*ret_model.output_layers, sep=',\n\t')

    return ret_model


if __name__ == "__main__":
    raise NotImplementedError("Test not implemented for 'Models.py'")  # TODO(marius): Implement tester
