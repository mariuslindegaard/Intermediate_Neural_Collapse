import torch
import torch.nn as nn
import torchvision.models

from typing import Dict, Hashable, Tuple, Union, List, Optional
from collections import OrderedDict
import warnings

from DatasetWrapper import DatasetWrapper
import utils


class ForwardHookedOutput(nn.Module):
    def __init__(self, base_model: nn.Module, output_layers_specification: Union[Tuple[Hashable], List[Hashable], bool],
                 base_model_specifier: str):
        # Init and store base model
        super().__init__()
        self.base_model = base_model
        self.base_model_specifier = base_model_specifier

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

        return self.output_layers

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


class SharedWeightMLP(MLP):
    def __init__(self, input_size: int, hidden_layers_widths: List[int], output_size: int,
                 use_bias: bool = True, use_softmax: bool = False, use_batch_norm: bool = True):
        super(MLP, self).__init__()


        layers = OrderedDict()

        layers['flatten'] = nn.Flatten()

        if len(hidden_layers_widths) == 0:
            layers['fc'] = nn.Linear(in_features=input_size, out_features=output_size, bias=use_bias)
        else:
            width = hidden_layers_widths[0]
            if width > input_size:
                warnings.warn(f"Width of NN ({width}) greater than the size of the input data ({input_size})."
                              f"Padding input to proper width.")
                layers['padding'] = torch.nn.ConstantPad1d(padding=(0, width-input_size), value=0)
            elif width < input_size:
                warnings.warn(f"Width of NN ({width}) SMALLER than the size of the input data ({input_size})."
                              f"Project input randomly to proper width.")
                layers['projection'] = nn.Linear(in_features=input_size, out_features=width, bias=False)
                layers['projection'].requires_grad = False

            assert not sum(map(lambda w: w != width, hidden_layers_widths)),\
                f"Not all hidden layers have the same width! Got {hidden_layers_widths}..."
            layers['block0'] = _MLPBlock(input_dim=width, output_dim=width,
                                         use_bias=use_bias, use_batch_norm=use_batch_norm)
            for idx in range(1, len(hidden_layers_widths)):
                block = _MLPBlock(input_dim=width, output_dim=width, use_bias=use_bias, use_batch_norm=use_batch_norm)
                # Set weights to be the same
                block.fc.weight = layers['block0'].fc.weight
                if use_bias:
                    block.fc.bias = layers['block0'].fc.bias
                if use_batch_norm:
                    block.bn.weight = layers['block0'].bn.weight
                    block.bn.bias = layers['block0'].bn.bias

                layers[f'block{idx+1}'] = block

            # Make the classification layer a simplex ETF
            layers['ETF'] = nn.Linear(in_features=width, out_features=output_size, bias=False)
            shape, c = layers['ETF'].weight.shape, output_size
            layers['ETF'].weight = torch.nn.Parameter(
                (c/(c-1))**(1/2) * (torch.eye(*shape) - (1/c) * torch.ones(*shape)),
                requires_grad=False)

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


class ConvNet(nn.Module):
    def __init__(self, input_channels: int, image_hw: torch.Tensor, output_size: int,
                 hidden_layers_channels: List[int],
                 downsampling_layers: Optional[List[Optional[int]]] = None,
                 use_bias: bool = False, use_softmax: bool = False, use_batch_norm: bool = True):
        super(ConvNet, self).__init__()
        if downsampling_layers is None:
            downsampling_layers = []

        # Make downsampling layers list match length of hidden channels
        self.downsampling_blocks = downsampling_layers + [None] * (len(hidden_layers_channels) - len(downsampling_layers))

        layers = OrderedDict()

        assert hidden_layers_channels, "Convnet must contain hidden layers before classification layer."

        self.preblock_image_hw = {'b0': image_hw}
        layers['b0'] = _ConvBlock(input_ch=input_channels, output_ch=hidden_layers_channels[0],
                                  downsample=self.downsampling_blocks[0],
                                  use_bias=use_bias, use_batch_norm=use_batch_norm, use_relu=False)

        downsampling = downsampling_layers[0]
        if downsampling is None:
            next_image_hw = self.preblock_image_hw[f'b0']
        else:
            next_image_hw = torch.div(self.preblock_image_hw[f'b0'], downsampling, rounding_mode='floor')

        # import pdb; pdb.set_trace()
        for idx, (in_ch, out_ch, downsampling) in enumerate(zip(hidden_layers_channels[:-1], hidden_layers_channels[1:], self.downsampling_blocks[1:])):
            self.preblock_image_hw = {f'b{idx+1}': next_image_hw}
            layers[f'b{idx+1}'] = _ConvBlock(input_ch=in_ch, output_ch=out_ch,
                                             downsample=downsampling,  # Is most often None
                                             use_bias=use_bias, use_batch_norm=use_batch_norm,
                                             # use_relu=downsampling is None or downsampling > 0
                                             )  # Don't use relu when specified downsampling is negative
            if downsampling is None:
                next_image_hw = self.preblock_image_hw[f'b{idx+1}']
            else:
                next_image_hw = torch.div(self.preblock_image_hw[f'b{idx+1}'], downsampling, rounding_mode='floor')

        layers['flatten'] = nn.Flatten()
        layers['fc'] = nn.Linear(hidden_layers_channels[-1]*next_image_hw.prod(dtype=int), output_size)

        if use_softmax:
            layers['softmax'] = nn.Softmax()

        self.model = nn.Sequential(
            layers
        )

        """
        self.num_input_channels = input_channels
        self.num_matrices = self.depth = settings.depth 
        self.bn = use_batch_norm

        layers = nn.Sequential()

        self.input_dimensions = [32]
        self.output_dimensions = [16]

        layers.append(nn.Conv2d(self.num_input_channels, hidden_layers_channels[0], 2, 2))
        if self.bn:
            layers.append(nn.BatchNorm2d(hidden_layers_channels[0]))
            self.input_dimensions.append(None)
            self.output_dimensions.append(None)

        self.input_dimensions += [16, None]
        self.output_dimensions += [8, None]
        layers.append(nn.Conv2d(hidden_layers_channels[0], hidden_layers_channels[1], 2, 2))
        if self.bn:
            layers.append(nn.BatchNorm2d(hidden_layers_channels[1]))
            self.input_dimensions.append(None)
            self.output_dimensions.append(None)
        layers.append(self.activation)

        for i in range(self.depth):
            self.input_dimensions += [8, None]
            self.output_dimensions += [8, None]

            layers.append(nn.Conv2d(self.width, self.width, 3, 1, 1))
            if self.bn:
                layers.append(nn.BatchNorm2d(self.width))
                self.input_dimensions.append(None)
                self.output_dimensions.append(None)
            layers.append(self.activation)

        self.layers = layers
        """

    def forward(self, x):
        out = self.model(x)
        return out


class _ConvBlock(nn.Module):
    def __init__(self, input_ch: int, output_ch: int, use_bias: bool = None, use_batch_norm: bool = False,
                 downsample: Optional[int] = None, use_relu: bool = True):
        super(_ConvBlock, self).__init__()

        # Handle downsampling
        # assert downsample > 0, f"Downsampling must be positive integer, but is {downsample}"
        self.use_relu = use_relu
        self.downsample = downsample

        if self.downsample:
            conv_params = dict(kernel_size=self.downsample, stride=self.downsample, padding=0)
        else:
            conv_params = dict(kernel_size=3, stride=1, padding=1)

        self.conv = nn.Conv2d(in_channels=input_ch, out_channels=output_ch, **conv_params, bias=use_bias)
        self.use_batch_norm = use_batch_norm
        if self.use_batch_norm:
            self.bn = nn.BatchNorm2d(num_features=output_ch)

        if self.use_relu:
            self.relu = nn.ReLU()

    def forward(self, x):
        x = self.conv(x)
        if self.use_batch_norm:
            x = self.bn(x)
        if self.use_relu:
            x = self.relu(x)
        return x

def get_model(model_cfg: Dict, datasetwrapper: DatasetWrapper):
    model_name: str = model_cfg['model-name'].lower()

    if model_name.startswith('resnet'):
        # Find specified resnet in torchvision.models
        assert hasattr(torchvision.models, model_name), f"Model type not supported: {model_name}" \
                                                        f"\nNo such resnet model in torchvision.models"
        base_model = getattr(torchvision.models, model_name)(pretrained=False)

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
            '_xlarge': [1024] * 20,
            '_huge': [2048] * 20,
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
        mlptype = MLP if '_sharedweight' not in model_name.lower() else SharedWeightMLP
        base_model = mlptype(
            input_size=datasetwrapper.input_batch_shape[1:].numel(),
            hidden_layers_widths=hidden_layer_sizes,
            output_size=datasetwrapper.num_classes,
            use_bias='_bias' in model_name,
            use_batch_norm='_nobn' not in model_name
        )

    elif model_name.startswith('convnet'):
        # Find specified convnet filter sizes
        suffix = model_name[7:]
        sizes = {
            '_small': [64]*7,
            '_default': [128]*10,
            '_deep': [128]*20,
            '_wide': [256]*10,
            '_huge': [512]*20,
        }
        sizes_match = ''
        for sizes_key in sizes.keys():
            if sizes_key in suffix:
                assert not sizes_match, f"Multiple matches for given ConvNet suffix: {sizes_key} and {sizes_match}."
                sizes_match = sizes_key
        if not sizes_match:
            sizes_match = '_default'
        hidden_layer_sizes = sizes[sizes_match]

        downsampling_layers = [2, 2]

        # Construct Model
        base_model = ConvNet(input_channels=datasetwrapper.input_batch_shape[1],
                             image_hw=torch.Tensor(tuple(datasetwrapper.input_batch_shape[2:])),
                             output_size=datasetwrapper.num_classes,
                             hidden_layers_channels=hidden_layer_sizes, downsampling_layers=downsampling_layers,
                             use_bias='_bias' in model_name, use_batch_norm='_nobn' not in model_name)
    else:
        raise NotImplementedError(f"Model type not supported: {model_name}")

    # Set hooked output layers
    out_layers = model_cfg['embedding-layers']
    # print(base_model)
    ret_model = ForwardHookedOutput(base_model, out_layers, model_name)
    # print("Tracking layers: ", end='\n\t')
    # print(*ret_model.output_layers, sep=',\n\t')

    return ret_model


if __name__ == "__main__":
    raise NotImplementedError("Test not implemented for 'Models.py'")  # TODO(marius): Implement tester
