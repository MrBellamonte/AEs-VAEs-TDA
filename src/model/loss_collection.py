from abc import abstractmethod, ABCMeta
from typing import Union

import torch
from torch.nn.modules.loss import L1Loss as L1Loss_torch


class Loss(metaclass=ABCMeta):
    def __init__(self):
        pass

    @abstractmethod
    def forward(self, input: torch.Tensor, target: Union[torch.Tensor, float]) -> torch.Tensor:
        pass

    @property
    @abstractmethod
    def fancy_name(self) -> str:
        pass


DEFAULT = {
    "l1loss"   : dict(reduction='mean'),
    "huberloss": dict(reduction='mean', delta=10),
    "hingeloss": dict(reduction='mean', penalty_type='linear', penalty_direction='positive')
}


class L1Loss(L1Loss_torch, Loss):
    """L1 loss"""

    fancy_name = "L1 loss"
    __slots__ = ['reduction']

    def __init__(self, *args, **kwargs):
        DEFAULT["l1loss"].update(kwargs)
        super().__init__(*args, **DEFAULT["l1loss"])

    def forward(self, input, target):
        return super().forward(input, target)


class PseudoHuberLoss(Loss):
    """Pseudo-Huber Loss"""
    __slots__ = ['reduction', 'delta']

    fancy_name = "Huber Loss"

    def __init__(self, reduction=DEFAULT["huberloss"]['reduction'],
                 delta=DEFAULT["huberloss"]['delta']):
        self.reduction = reduction
        self.delta = delta

    def forward(self, input, target):
        if self.reduction == 'mean':
            return torch.mean(self.delta**2*(torch.sqrt(1+((input-target)/self.delta)**2)-1))
        elif self.reduction == None:
            return torch.sum(self.delta**2*(torch.sqrt(1+((input-target)/self.delta)**2)-1))
        else:
            raise ValueError


class HingeLoss(Loss):
    """Hinge Loss"""
    __slots__ = ['penalty_type', 'reduction', 'penalty_direction']

    fancy_name = "Hinge Loss"

    def __init__(self, reduction=DEFAULT["hingeloss"]['reduction'],
                 penalty_type=DEFAULT["hingeloss"]['penalty_type'],
                 penalty_direction=DEFAULT["hingeloss"]['penalty_direction']):
        self.reduction = reduction
        self.penalty_type = penalty_type
        self.penalty_direction = penalty_direction

    def forward(self, input, target):
        if self.penalty_direction == 'positive':
            if self.penalty_type == 'linear':
                temp = torch.clamp((input-target), min=0)
            elif self.penalty_type == 'squared':
                temp = torch.clamp((input-target), min=0)**2
            else:
                raise ValueError
        elif self.penalty_direction == 'negative':
            if self.penalty_type == 'linear':
                temp = torch.clamp((input-target), max=0).abs()
            elif self.penalty_type == 'squared':
                temp = torch.clamp((input-target), max=0)**2
            else:
                raise ValueError
        else:
            raise ValueError

        if self.reduction == 'mean':
            return torch.mean(temp)
        elif self.reduction == None:
            return torch.sum(temp)
        else:
            raise ValueError
