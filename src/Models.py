import torch
import torch.nn as nn
import torchvision.models as models

from typing import Dict, Optional, Iterable, Hashable
from collections import OrderedDict


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


def get_model(model_cfg: Dict):
    device = 'cuda' if torch.cuda.is_available() else 'cpu'
    model_name = model_cfg['model-name'].lower()

    if model_name == 'resnet18':
        base_model = models.resnet18(pretrained=False)  # TODO(marius): Make more models available than resnet18
        base_model.fc = nn.Linear(in_features=base_model.fc.in_features, out_features=10)  # TODO(marius): Make use num classes
        base_model.to(device)
    else:
        assert model_cfg['model-name'] == 'resnet18', "Resnet18 is always used for now. TODO to implement other models."
        raise NotImplementedError

    out_layers = model_cfg['embedding_layers']
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

