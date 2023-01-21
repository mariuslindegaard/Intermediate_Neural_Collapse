import torch
import torch.nn
import torch.nn.functional

from typing import Dict

import Models


class OptimizerWrapper(torch.nn.Module):
    criterion: torch.nn.Module
    optimizer: torch.optim.Optimizer  # <- Optimizer
    lr_scheduler: object  # <- learning rate scheduler
    max_epochs: int
    num_warmup_epochs: int
    # epoch: int

    _criterions = dict(
        mseloss=torch.nn.MSELoss,
        celoss=torch.nn.CrossEntropyLoss,
    )

    def __init__(self, model: Models.ForwardHookedOutput, optimizer_cfg: Dict):
        super().__init__()
        self.criterion = self._criterions[optimizer_cfg['loss'].lower()]()

        self.optimizer = torch.optim.SGD(model.parameters(),
                                         lr=optimizer_cfg['lr'],
                                         momentum=optimizer_cfg['momentum'],
                                         weight_decay=optimizer_cfg['weight-decay'])

        lr_decay_steps = optimizer_cfg['lr-decay-steps']
        if isinstance(lr_decay_steps, int):
            epochs_lr_decay = [i * optimizer_cfg['epochs'] // optimizer_cfg['lr-decay-steps']
                               for i in range(1, optimizer_cfg['lr-decay-steps'])]
        elif isinstance(lr_decay_steps, list):
            epochs_lr_decay = lr_decay_steps
        else:
            raise TypeError("Optimizer param lr-decay-steps must be list of epochs or int number of intervals.")

        self.lr_scheduler = torch.optim.lr_scheduler.MultiStepLR(self.optimizer,
                                                                 milestones=epochs_lr_decay,
                                                                 gamma=optimizer_cfg['lr-decay'])

        self.max_epochs = int(optimizer_cfg['epochs'])

        self.num_warmup_epochs = int(optimizer_cfg.get('warmup_epochs', 0))

