import torch
import torch.nn as nn
import torchvision.models as models

from typing import Dict, Optional, Iterable, Hashable, Tuple
from collections import OrderedDict

from DatasetWrapper import DatasetWrapper


class ForwardHookedOutput(nn.Module):
    def __init__(self, base_model: nn.Module, output_layers: Optional[Iterable[Hashable]], *args):
        # Init and store base model
        super().__init__(*args)
        self.base_model = base_model

        # Output hooks
        self.output_layers = tuple(output_layers)
        self.fwd_hooks = []
        self.hook_out = OrderedDict()
        self._module_to_layer_name = {}  # Mapping of modules to layername

        # TODO(marius): Allow accessing nested layers!!!
        # Register hooks
        for i, l in enumerate(list(self.base_model._modules.keys())):
            if l in self.output_layers:
                layer = getattr(self.base_model, l)
                self._module_to_layer_name[layer] = l
                self.fwd_hooks.append(
                    layer.register_forward_hook(self.hook)
                )

    def hook(self, module, inputs, outputs):
        layer_name = self._module_to_layer_name[module]
        self.hook_out[layer_name] = outputs

    def forward(self, x):
        out = self.base_model(x)
        return out, self.hook_out


class MLP(nn.Module):
    def __init__(self, input_size: int, hidden_layers_widths: Iterable[int], output_size: int,
                 use_bias: bool = True, use_softmax: bool = False):
        super(MLP, self).__init__()


        self.model = nn.Sequential(
            nn.Flatten()
        )
        if len(hidden_layers_widths) == 0:
            self.model.append(nn.Linear(in_features=input_size, out_features=output_size, bias=use_bias))
        else:
            self.model.append(nn.Linear(in_features=input_size, out_features=hidden_layers_widths[0], bias=use_bias))
            self.model.append(nn.ReLU())
            for idx, (in_size, out_size) in enumerate(zip(hidden_layers_widths[:-1], hidden_layers_widths[1:])):
                self.model.append(nn.Linear(in_features=in_size, out_features=out_size, bias=use_bias))
                self.model.append(nn.ReLU())
            self.model.append(nn.Linear(in_features=hidden_layers_widths[-1], out_features=output_size))

        if use_softmax:
            self.model.append(nn.Softmax())

    def forward(self, x):
        out = self.model(x)
        return out


def get_model(model_cfg: Dict, datasetwrapper: DatasetWrapper):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = model_cfg['model-name'].lower()

    if model_name == 'resnet18':
        base_model = models.resnet18(pretrained=False)
        base_model.fc = nn.Linear(in_features=base_model.fc.in_features, out_features=datasetwrapper.num_classes)
        base_model.to(device)
    elif model_name == 'mlp':
        hidden_layer_sizes = [128, 128, 64, 64]
        base_model = MLP(
            input_size=datasetwrapper.input_batch_shape[1:].numel(),
            hidden_layers_widths=hidden_layer_sizes,
            output_size=datasetwrapper.num_classes
        )
        base_model.to(device)
    else:
        assert model_cfg['model-name'].lower() in ('resnet18', 'mlp'),\
            f"Model type not supported: {model_cfg['model-name']}"
        raise NotImplementedError

    out_layers = model_cfg['embedding_layers']  # TODO(marius): Make support "True" for all layers
    print(base_model)
    ret_model = ForwardHookedOutput(base_model, out_layers).to(device)
    return ret_model

def get_resnet18_model(model_cfg: Dict):
    # TODO(marius): Add support for mnist single channel input data
    # base_model.fc = nn.Linear(in_features=512, out_features=10, bias=True)
    # base_model.to(device)

    # out_layers = {f'layer{i}': 1 for i in range(3, 5)}
    # out_layers = model_cfg['embedding_layers']
    # ret_model = ForwardHookedOutput(base_model, out_layers).to(device)
    raise NotImplementedError()
    return ret_model


if __name__ == "__main__":
    _ret_model = get_resnet18_model(dict(
        embedding_layers={'layer3': 1, 'layer4': 1}
    ))
    print(_ret_model)
    pass

